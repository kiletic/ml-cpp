set_languages("c++20")
set_warnings("everything")
add_includedirs("include")

add_rules("plugin.compile_commands.autoupdate")

add_rules("mode.release", "mode.debug")
if is_mode("debug") then
  add_cxxflags("-fsanitize=address", "-fsanitize=undefined")
end

add_requires("gtest", { configs = { main = true }})

target("value")
  set_kind("static")
  add_files("src/value.cpp")

target("test")
  set_default(false)
  set_kind("binary")
  add_packages("gtest", "gtest_main")
  add_files("tests/*.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end

target("main")
  set_kind("binary")
  add_deps("value")
  add_files("src/main.cpp")
  if is_mode("debug") then
    add_ldflags("-fsanitize=address", "-fsanitize=undefined")
  end
