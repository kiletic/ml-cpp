#include <iostream>
#include <util/util.h>

std::vector<std::string> split(std::string const &str, char delim) {
  std::vector<std::string> res;
  size_t start_pos = 0;
  size_t current_pos = 0;
  size_t next_pos = 0;
  while ((next_pos = str.find(delim, current_pos)) != std::string::npos) {
    if (str[start_pos] != '"' || (str[start_pos] == '"' && str[next_pos - 1] == '"')) {
      res.push_back(str.substr(start_pos, next_pos - start_pos));
      start_pos = next_pos + 1;
    }
    current_pos = next_pos + 1;
  }
  if (start_pos != str.size())
    res.push_back(str.substr(start_pos, str.size() - start_pos));
  return res;
}
