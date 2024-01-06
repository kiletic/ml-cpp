set_languages("c++20")
set_warnings("everything")
add_includedirs("include")

add_rules("plugin.compile_commands.autoupdate")

add_rules("mode.release", "mode.debug")
if is_mode("debug") then
  add_cxxflags("-fsanitize=address", "-fsanitize=undefined")
end

add_requires("gtest", { configs = { main = true }})
add_requires("libtorch")

target("value")
  set_kind("static")
  add_files("src/value.cpp")

target("all")
  set_kind("static")
  add_files("src/**.cpp")

target("test")
  set_default(false)
  set_kind("binary")
  add_packages("gtest", "gtest_main", "libtorch")
  add_deps("value")
  add_files("tests/*.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end

target("example_xor")
  set_default(false)
  set_kind("binary")
  add_deps("value")
  add_files("examples/xor.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end

target("example_xor_nnet")
  set_default(false)
  set_kind("binary")
  add_deps("all")
  add_files("examples/xor_nnet.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end

target("kaggle")
  set_kind("binary")
  add_deps("all")
  add_files("examples/kaggle/main.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end

target("main")
  set_kind("binary")
  add_deps("all")
  add_files("examples/main.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end
