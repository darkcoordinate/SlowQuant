#include <iomanip>
#include <iostream>
#include <map>
#include <string>

int main(){
  std::map<std::string, float> heights;
  heights.insert({"John", 1.8});
  heights.insert({"Jane", 1.6});
  heights.insert({"Bob", 1.7});

  std::map<std::string, float> heights2;
  heights2.insert({"John", 1.8});
  heights2.insert({"Jane", 1.6});
  heights2.insert({"Bobb", 1.7});
  std::cout<<heights2.size()<<std::endl;
  for (const auto& [name, height] : heights) {
    std::cout << name << ": " << height << "m\n";
    heights2[name] += height;
  }
  std::cout<<" fij"<<std::endl;  
  std::cout<<heights2.size()<<std::endl;
  for (const auto& [name, height] : heights2) {
    std::cout << name << ": " << height << "m\n";
  }  
}