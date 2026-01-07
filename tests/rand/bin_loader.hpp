#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class RandLoader {
public:
  template <typename Scalar = float>
  static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> loadMatrix(const std::string &path, int max_rows = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + path);
    }

    std::int32_t rows = 0;
    std::int32_t cols = 0;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    if (!file) {
      throw std::runtime_error("Failed to read header: " + path);
    }
    if (rows <= 0 || cols <= 0) {
      throw std::runtime_error("Invalid matrix shape in: " + path);
    }

    std::int32_t count = rows;
    if (max_rows > 0 && max_rows < rows) {
      count = max_rows;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(cols, count);
    std::vector<float> row(static_cast<size_t>(cols));

    for (std::int32_t r = 0; r < count; ++r) {
      const auto row_bytes = static_cast<std::streamsize>(row.size() * sizeof(float));

      file.read(reinterpret_cast<char *>(row.data()), row_bytes);
      if (!file) throw std::runtime_error("Failed to read matrix data: " + path);

      Eigen::Map<const Eigen::VectorXf> row_vec(row.data(), cols);
      out.col(r) = row_vec.template cast<Scalar>();
    }

    std::cout << "Loaded " << count << " rows from " << path << " (" << cols << " cols) [Transposed]" << std::endl;
    return out;
  }
};