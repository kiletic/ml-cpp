#include <gtest/gtest.h>

TEST(TestSuite, TestFunction) {
  EXPECT_EQ(2, 2);
  ASSERT_TRUE(false);
}

TEST(TestSuite, TestFunction2) {
  EXPECT_EQ(2, 3);
}
