set_languages("c++20")
set_warnings("everything")
add_includedirs("include")

add_rules("plugin.compile_commands.autoupdate")

add_rules("mode.release", "mode.debug")
if is_mode("debug") then
  set_policy("build.sanitizer.address", true)
  set_policy("build.sanitizer.undefined", true)
else
  -- so it doesn't reinstall packages with asan enabled...
  -- TODO: how to resolve this? 
  add_requires("gtest", { configs = { main = true }})
  add_requires("libtorch")
end

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

target("example_xor")
  set_default(false)
  set_kind("binary")
  add_deps("value")
  add_files("examples/xor.cpp")

target("example_xor_nnet")
  set_default(false)
  set_kind("binary")
  add_deps("all")
  add_files("examples/xor_nnet.cpp")

target("example_xor_torch")
  set_default(false)
  set_kind("binary")
  add_packages("libtorch")
  add_deps("all")
  add_files("examples/xor_torch.cpp")

target("kaggle")
  set_kind("binary")
  add_deps("all")
  add_files("examples/kaggle/main.cpp")

target("main")
  set_kind("binary")
  add_deps("all")
  add_files("examples/main.cpp")
