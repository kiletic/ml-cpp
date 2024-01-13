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

includes("examples")
includes("tests")

target("value")
  set_kind("static")
  add_files("src/value.cpp")

target("all")
  set_kind("static")
  add_files("src/**.cpp")

