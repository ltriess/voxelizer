#include "voxelize_utils.h"

#include <rv/string_utils.h>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>


using namespace rv;

std::vector<std::string> parseDictionary(std::string str) {
  std::vector<std::string> tokens;
  str = trim(str);
  if (str[0] != '{' || str[str.size() - 1] != '}') {
    std::cerr << "Parser Error: " << str << " is not a valid dictionary token!" << std::endl;
    return tokens;
  }

  tokens = split(str.substr(1, str.size() - 2), ":");

  return tokens;
}

template <class T>
std::vector<T> parseList(std::string str) {
  str = trim(str);
  std::vector<T> list;

  str = trim(str);
  if (str[0] != '[' || str[str.size() - 1] != ']') {
    std::cerr << "Parser Error: " << str << " is not a valid list token!" << std::endl;
    return list;
  }

  auto entry_tokens = split(str.substr(1, str.size() - 2), ",");

  for (const auto& token : entry_tokens) {
    T value = boost::lexical_cast<T>(trim(token));
    list.push_back(value);
  }

  return list;
}

template <>
std::vector<std::string> parseList(std::string str) {
  str = trim(str);
  if (str[0] != '[' || str[str.size() - 1] != ']') {
    std::cerr << "Parser Error: " << str << " is not a valid list token!" << std::endl;
  }

  std::vector<std::string> list;
  auto entry_tokens = split(str.substr(1, str.size() - 2), ",");

  for (uint32_t i = 0; i < entry_tokens.size(); ++i) {
    std::string token = entry_tokens[i];

    if (token.find("[") != std::string::npos) {
      if (token.find("]", token.find("[")) == std::string::npos) {
        // found a nested unterminated list token
        std::string next_token;
        do {
          if (i >= entry_tokens.size()) break;
          next_token = entry_tokens[i + 1];
          token += "," + next_token;
          ++i;
        } while (next_token.find("]") == std::string::npos);
      }
    }
    list.push_back(token);
  }

  return list;
}

Config parseConfiguration(const std::string& filename) {
  Config config{};
  std::ifstream in(filename);

  if (!in.is_open()) return config;

  std::string line;
  in.peek();
  while (in.good() && !in.eof()) {
    std::getline(in, line);

    if (trim(line)[0] == '#') continue;  // ignore comments.

    auto tokens = split(line, ":");
    if (tokens.size() < 2) continue;
    if (tokens.size() > 2) {
      for (uint32_t i = 2; i < tokens.size(); ++i) {
        tokens[1] += ":" + tokens[i];
      }
      tokens.resize(2);
    }

    if (tokens[0] == "max scans") {
      config.maxNumScans = boost::lexical_cast<uint32_t>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "max range") {
      config.maxRange = boost::lexical_cast<float>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "voxel size") {
      config.voxelSize = boost::lexical_cast<float>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "min range") {
      config.minRange = boost::lexical_cast<float>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "prior scans") {
      config.priorScans = boost::lexical_cast<uint32_t>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "past scans") {
      config.pastScans = boost::lexical_cast<uint32_t>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "past distance") {
      config.pastScans = boost::lexical_cast<float>(trim(tokens[1]));
      continue;
    }

    if (tokens[0] == "stride num") {
      config.stride_num = boost::lexical_cast<uint32_t>(trim(tokens[1]));
      continue;
    }
    if (tokens[0] == "stride distance") {
      config.stride_distance = boost::lexical_cast<float>(trim(tokens[1]));
      continue;
    }

    if (tokens[0] == "min extent") {
      auto coords = parseList<float>(tokens[1]);
      config.minExtent = Eigen::Vector4f(coords[0], coords[1], coords[2], 1.0f);
      continue;
    }

    if (tokens[0] == "max extent") {
      auto coords = parseList<float>(tokens[1]);
      config.maxExtent = Eigen::Vector4f(coords[0], coords[1], coords[2], 1.0f);

      continue;
    }

    if (tokens[0] == "ignore") {
      config.filteredLabels = parseList<uint32_t>(tokens[1]);

      continue;
    }

    if (tokens[0] == "join") {
      auto join_tokens = parseList<std::string>(trim(tokens[1]));

      for (const auto& token : join_tokens) {
        auto mapping = parseDictionary(token);
        uint32_t label = boost::lexical_cast<uint32_t>(trim(mapping[0]));
        config.joinedLabels[label] = parseList<uint32_t>(trim(mapping[1]));
      }

      continue;
    }

    std::cout << "unknown parameter: " << tokens[0] << std::endl;
  }

  in.close();

  config.ignored_label_set.insert(config.filteredLabels.cbegin(), config.filteredLabels.cend());

  return config;
}

void fillVoxelGrid(const Eigen::Matrix4f& anchor_pose, const std::vector<PointcloudPtr>& points,
                   const std::vector<LabelsPtr>& labels, VoxelGrid& grid, const Config& config) {
  std::map<uint32_t, uint32_t> mappedLabels;  // replace key with value.
  for (auto joins : config.joinedLabels) {
    for (auto label : joins.second) {
      mappedLabels[label] = joins.first;
    }
  }

  for (uint32_t t = 0; t < points.size(); ++t) {
    Eigen::Matrix4f ap = anchor_pose.inverse() * points[t]->pose;

    const auto scan_index = points[t]->getIdx();

    grid.addTransform(scan_index, ap);

    for (uint32_t i = 0; i < points[t]->points.size(); ++i) {
      const Point3f& pp = points[t]->points[i];

      float range = Eigen::Vector3f(pp.x, pp.y, pp.z).norm();
      if (range < config.minRange || range > config.maxRange) continue;
      bool is_car_point = (config.hidecar && pp.x < 3.0 && pp.x > -2.0 && std::abs(pp.y) < 2.0);
      if (is_car_point) continue;

      // transform individual point into anchor coordinate system
      Eigen::Vector4f p = ap * Eigen::Vector4f(pp.x, pp.y, pp.z, 1);

      uint32_t label = (*labels[t])[i];
      if (mappedLabels.find(label) != mappedLabels.end()) {
        label = mappedLabels[label];
      }

      // Todo: static points ignore but-not-on-first-frame hack
      if (grid.ignoresDisabled() || config.ignored_label_set.find(label) == config.ignored_label_set.cend()){
        grid.insert(p, (*labels[t])[i], scan_index);
      }

//      if (std::find(config.filteredLabels.begin(), config.filteredLabels.end(), label) == config.filteredLabels.end()) {
//        grid.insert(p, (*labels[t])[i], scan_index);
//      }
    }
  }
}

template <typename T>
std::vector<uint8_t> pack(const std::vector<T>& vec) {
  std::vector<uint8_t> packed(vec.size() / 8);

  for (uint32_t i = 0; i < vec.size(); i += 8) {
    packed[i / 8] = (vec[i] > 0) << 7 | (vec[i + 1] > 0) << 6 | (vec[i + 2] > 0) << 5 | (vec[i + 3] > 0) << 4 |
                    (vec[i + 4] > 0) << 3 | (vec[i + 5] > 0) << 2 | (vec[i + 6] > 0) << 1 | (vec[i + 7] > 0);
    ;
  }

  return packed;
}

void saveVoxelGrid(const VoxelGrid& grid, const std::string& directory, const std::string& basename,
                   const std::string& mode) {
  uint32_t Nx = grid.size(0);
  uint32_t Ny = grid.size(1);
  uint32_t Nz = grid.size(2);

  size_t numElements = grid.num_elements();
  std::vector<uint16_t> outputLabels(numElements, 0);
  std::vector<uint32_t> outputTensorOccluded(numElements, 0);
  std::vector<uint32_t> outputTensorInvalid(numElements, 0);
  std::vector<uint32_t> outputTensorDynamicOcclusions(numElements, 0);

  int32_t counter = 0;
  for (uint32_t x = 0; x < Nx; ++x) {
    for (uint32_t y = 0; y < Ny; ++y) {
      for (uint32_t z = 0; z < Nz; ++z) {
        const VoxelGrid::Voxel& v = grid(x, y, z);

        uint32_t isOccluded = (uint32_t)grid.isOccluded(x, y, z);

        uint32_t maxCount = 0;
        uint32_t maxLabel = 0;

        for (auto it = v.labels.begin(); it != v.labels.end(); ++it) {
          if (it->second > maxCount) {
            maxCount = it->second;
            maxLabel = it->first;
          }
        }

        // Write maxLabel appropriately to file.
        assert(counter < numElements);
        outputLabels[counter] = maxLabel;
        outputTensorOccluded[counter] = isOccluded;
        outputTensorInvalid[counter] = (uint32_t)grid.isInvalid(x, y, z);
        counter = counter + 1;
      }
    }
  }

  if (mode == "target") {
    // for target we just generate label, invalid, occluded.
    {
      std::string output_filename = directory + "/" + basename + ".label";

      std::ofstream out(output_filename.c_str());
      out.write((const char*)&outputLabels[0], outputLabels.size() * sizeof(uint16_t));
      out.close();
    }

    {
      std::string output_filename = directory + "/" + basename + ".occluded";

      std::ofstream out(output_filename.c_str());
      std::vector<uint8_t> packed = pack(outputTensorOccluded);
      out.write((const char*)&packed[0], packed.size() * sizeof(uint8_t));
      out.close();
    }

    {
      std::string output_filename = directory + "/" + basename + ".invalid";

      std::ofstream out(output_filename.c_str());
      std::vector<uint8_t> packed = pack(outputTensorInvalid);
      out.write((const char*)&packed[0], packed.size() * sizeof(uint8_t));
      out.close();
    }

    // write accumulated points
    saveAccumulatedPoints(grid, directory + "/" + basename + "_points.tfrecord");

  } else {

    std::unordered_set<uint16_t> dynamic_labels{252, 253, 254, 255, 256, 257, 258, 259};

    int32_t counter = 0;
    for (uint32_t x = 0; x < Nx; ++x) {
      for (uint32_t y = 0; y < Ny; ++y) {
        for (uint32_t z = 0; z < Nz; ++z) {
          
          const auto idx = grid.index(x, y, z);

          const int32_t occlusion_index = grid.getOcclusion(idx);
          if (occlusion_index >= 0) {
            assert(occlusion_index < numElements);
            const uint16_t occlusion_label = outputLabels[occlusion_index];

            outputTensorDynamicOcclusions[counter] =
                static_cast<uint32_t>(dynamic_labels.find(occlusion_label) != dynamic_labels.end());
          }

          assert(counter < numElements);
          ++counter;
        }
      }
    }

    // for input we just generate the ".bin" file.
    {
      std::string output_filename = directory + "/" + basename + ".bin";

      std::ofstream out(output_filename.c_str());
      std::vector<uint8_t> packed = pack(outputLabels);
      out.write((const char*)&packed[0], packed.size() * sizeof(uint8_t));
      out.close();
    }

    {
      std::string output_filename = directory + "/" + basename + ".dynamic_occlusion";

      std::ofstream out(output_filename.c_str());
      std::vector<uint8_t> packed = pack(outputTensorDynamicOcclusions);
      out.write((const char*)&packed[0], packed.size() * sizeof(uint8_t));
      out.close();
    }
  }
}

void add_transform_matrices(
    ::tensorflow::Features *features,
    const std::vector<Eigen::Matrix4f> &transform_matrices,
    const std::string &key) {
  // Eigen has column major matrix storage. Write float values as row major!
  auto *float_list = new tensorflow::FloatList{};
  for (const auto &matrix : transform_matrices) {
    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> m_row_major(matrix);
    for (uint32_t i = 0;
         i < m_row_major.size(); i++) {
      float_list->add_value(m_row_major.data()[i]);
    }
  }

  ::tensorflow::Feature feature{};
  feature.set_allocated_float_list(float_list);
  assert(feature.has_float_list());

  google::protobuf::MapPair<::std::string, ::tensorflow::Feature> map_entry(
      key, feature);
  features->mutable_feature()->insert(map_entry);
}

void saveAccumulatedPoints(const VoxelGrid& grid, const std::string& filename) {
  const VoxelGrid::PointMap& points = grid.getPointMap();
  const VoxelGrid::TransformMap& tfs = grid.getTransformMap();

  if (points.empty()) {
    return;
  }

  // get all keys = cloud idx
  std::vector<uint32_t> keys{};
  keys.reserve(points.size());
  std::transform(points.cbegin(), points.cend(), std::back_inserter(keys),
      [](const VoxelGrid::PointMap::value_type& pair){return pair.first;});

  std::sort(keys.begin(), keys.end());
  uint32_t point_counter = 0;

  std::cout << "Minimal point index: " << keys.front() << "\nMaximal point index: " << keys.back() << std::endl;

  tensorflow::Example example{};
  auto *features = new ::tensorflow::Features{};

  std::vector<Eigen::Matrix4f> tfs_vector{};
  for (auto idx = keys.front(); idx <= keys.back(); ++idx) {
    const auto it = tfs.find(idx);
    if (it != tfs.cend()) {
      tfs_vector.emplace_back(it->second);
    }
    else {
      tfs_vector.emplace_back(Eigen::Matrix4f::Constant(std::numeric_limits<float>::quiet_NaN()));
    }
  }
  add_transform_matrices(features, tfs_vector, "transforms");

  // Eigen has column major matrix storage. Write float values as row major!
  auto *float_list = new tensorflow::FloatList{};
  auto *split_list = new tensorflow::Int64List{};
  auto *label_bytes = new tensorflow::BytesList{};

  auto *min_max = new tensorflow::Int64List{};
  min_max->add_value(keys.front());
  min_max->add_value(keys.back() + 1);

  std::string label_string{};
  label_string.reserve(grid.getPointCount() * 2);
  uint32_t counter = 0;

  for (auto idx = keys.front(); idx <= keys.back(); ++idx) {

    const auto it = points.find(idx);
    if (it != points.cend()) {
      const auto& point_list = it->second;

      for (const auto& p : point_list) {
        float_list->add_value(p.point[0]);
        float_list->add_value(p.point[1]);
        float_list->add_value(p.point[2]);
        // label are lower 2 bytes of uint32 label.
        const uint16_t l = p.label & 0xFFFF;
        label_string.push_back(static_cast<char>(l & 0x00FF));
        label_string.push_back(static_cast<char>(l >> 8));
        ++counter;
      }

      point_counter += point_list.size();
    }
    split_list->add_value(static_cast<int64_t>(point_counter));

  }

  assert(counter == grid.getPointCount());
  label_bytes->add_value(label_string.c_str(), grid.getPointCount() * 2);

  {
    ::tensorflow::Feature feature{};
    feature.set_allocated_float_list(float_list);
    assert(feature.has_float_list());

    google::protobuf::MapPair<::std::string, ::tensorflow::Feature> map_entry("points", feature);
    features->mutable_feature()->insert(map_entry);
  }
  {
    ::tensorflow::Feature feature{};
    feature.set_allocated_int64_list(split_list);
    assert(feature.has_int64_list());

    google::protobuf::MapPair<::std::string, ::tensorflow::Feature> map_entry("splits", feature);
    features->mutable_feature()->insert(map_entry);
  }
  {
    ::tensorflow::Feature feature{};
    feature.set_allocated_bytes_list(label_bytes);
    assert(feature.has_bytes_list());

    google::protobuf::MapPair<::std::string, ::tensorflow::Feature> map_entry("labels", feature);
    features->mutable_feature()->insert(map_entry);
  }
  {
    ::tensorflow::Feature feature{};
    feature.set_allocated_int64_list(min_max);
    assert(feature.has_int64_list());

    google::protobuf::MapPair<::std::string, ::tensorflow::Feature> map_entry("scan_idx_min_max", feature);
    features->mutable_feature()->insert(map_entry);
  }

  example.set_allocated_features(features);
  std::fstream output(filename.c_str(),
                      std::ios::out | std::ios::trunc | std::ios::binary);
  if (!example.SerializeToOstream(&output)) {
    std::cerr << "Failed to write protobuf." << std::endl;
  }

}
