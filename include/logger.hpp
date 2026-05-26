#pragma once

#include <format>
#include <iostream>
#include <print>

namespace genivf {
namespace log {

enum class Level
{
    NONE = 0,
    INFO = 1,
    DEBUG = 2
};

inline Level g_level = Level::INFO;

inline void
set_level(Level level) noexcept
{
    g_level = level;
}

template<typename... Args>
inline void
info(std::format_string<Args...> fmt, Args&&... args)
{
    if (g_level >= Level::INFO) {
        std::print(std::clog, "[GENIVF INFO] ");
        std::println(std::clog, fmt, std::forward<Args>(args)...);
    }
}

template<typename... Args>
inline void
debug(std::format_string<Args...> fmt, Args&&... args)
{
    if (g_level >= Level::DEBUG) {
        std::print(std::clog, "[GENIVF DEBUG] ");
        std::println(std::clog, fmt, std::forward<Args>(args)...);
    }
}

} // namespace log
} // namespace genivf
