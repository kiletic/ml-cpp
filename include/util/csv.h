#include <string>
#include <unordered_map>
#include <vector>

struct CsvFile {
  std::vector<std::unordered_map<std::string, std::string>> rows;
  std::vector<std::string> columns;
  
  CsvFile() {}
  CsvFile(std::string const &path);

  std::vector<std::string> get_columns(); 
  size_t len();
};
