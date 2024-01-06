#include <fstream>
#include <iostream>
#include "util/csv.h"
#include "util/util.h"

CsvFile::CsvFile(std::string const &path) {
  std::ifstream file(path);
  
  std::string line;
  std::getline(file, line);
  this->columns = split(line, ',');

  while (std::getline(file, line)) {
    auto row_info = split(line, ',');

    std::unordered_map<std::string, std::string> row;
    for (size_t i = 0; i < this->columns.size(); i++) 
      row[this->columns[i]] = row_info[i];

    rows.push_back(std::move(row));
  }
}

std::vector<std::string> CsvFile::get_columns() {
  return this->columns;
} 

size_t CsvFile::len() {
  return this->rows.size();
}
