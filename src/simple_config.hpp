#pragma once

#include <cctype>
#include <fstream>
#include <string>
#include <unordered_map>

class SimpleConfig {
public:
  static SimpleConfig load(const std::string &path) {
    SimpleConfig cfg;
    std::ifstream in(path);
    if (!in) {
      return cfg;
    }
    cfg.loaded_ = true;

    std::string line;
    while (std::getline(in, line)) {
      std::string trimmed = trim(line);
      if (trimmed.empty() || trimmed.front() == '#') {
        continue;
      }
      std::size_t sep = trimmed.find(':');
      if (sep == std::string::npos) {
        continue;
      }
      std::string key = trim(trimmed.substr(0, sep));
      std::string value = trim(trimmed.substr(sep + 1));
      if (value.size() >= 2) {
        char first = value.front();
        char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
          value = value.substr(1, value.size() - 2);
        }
      }
      if (!key.empty()) {
        cfg.values_[key] = value;
      }
    }
    return cfg;
  }

  bool loaded() const { return loaded_; }

  int getInt(const std::string &key, int fallback) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
      return fallback;
    }
    try {
      return std::stoi(it->second);
    } catch (...) {
      return fallback;
    }
  }

  double getDouble(const std::string &key, double fallback) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
      return fallback;
    }
    try {
      return std::stod(it->second);
    } catch (...) {
      return fallback;
    }
  }

  std::string getString(const std::string &key, const std::string &fallback) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
      return fallback;
    }
    return it->second;
  }

private:
  static std::string trim(const std::string &input) {
    std::size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
      ++start;
    }
    if (start == input.size()) {
      return "";
    }
    std::size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
      --end;
    }
    return input.substr(start, end - start);
  }

  bool loaded_ = false;
  std::unordered_map<std::string, std::string> values_;
};
