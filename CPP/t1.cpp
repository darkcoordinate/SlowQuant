#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main() {
  std::map<std::vector<std::tuple<int, bool>>, double> heights;
  heights.insert({{std::make_tuple(1, true)}, 1.8});
  heights.insert({{std::make_tuple(2, false)}, 1.6});
  heights.insert({{std::make_tuple(3, true)}, 1.7});
  for (const auto &[key, value] : heights) {
    std::cout << std::get<0>(key[0]) << " " << std::get<1>(key[0]) << ": "
              << value << "m\n";
  }

  std::vector<std::tuple<int, bool>> key = {{1, true}};
  std::vector<std::tuple<int, bool>> key2;
  std::vector<std::tuple<int, bool>> key3;

  heights[key2] = 1.5;
  auto it = heights.find(key3);
  if (it != heights.end()) {
    std::cout << "Found " << heights[key3] << "\n";
  } else {
    std::cout << "Not found\n";
  }
  std::cout << heights[key] << "\n";
  std::cout << heights[key2] << "\n";

  for (const auto &[key, value] : heights) {
    std::cout << value << "m\n";
  }

  std::vector<std::tuple<int, bool>> v1 = {{1, true}, {2, false}};
  std::vector<std::tuple<int, bool>> v2 = {{1, true}, {2, false}};

  if (v1 == v2) {
    std::cout << "The vectors are identical." << std::endl;
  } else {
    std::cout << "The vectors are not identical." << std::endl;
  }
}