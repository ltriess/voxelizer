#ifndef SRC_DATA_POINTCLOUD_H_
#define SRC_DATA_POINTCLOUD_H_

#include <eigen3/Eigen/Dense>
#include "geometry.h"
#include <vector>

/** \brief a laserscan with possibly remission.
 *
 *  \author behley
 */

class Laserscan {
 public:

  Laserscan() = delete;
  Laserscan(uint32_t idx)
  : idx_{idx} {
  }

  void clear() {
    points.clear();
    remissions.clear();
  }
  uint32_t size() const { return points.size(); }
  bool hasRemissions() const { return (points.size() > 0) && (points.size() == remissions.size()); }

  Eigen::Matrix4f pose;
  std::vector<Point3f> points;
  std::vector<float> remissions;

  uint32_t getIdx() const { return idx_; }

 private:
  uint32_t idx_;
};

#endif /* SRC_DATA_POINTCLOUD_H_ */
