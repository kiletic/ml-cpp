set_languages("c++20")
set_warnings("everything")
add_includedirs("include")

target("value")
  set_kind("static")
  add_files("src/value.cpp")

target("main")
  set_kind("binary")
  add_deps("value")
  add_files("src/main.cpp")
