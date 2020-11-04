#pragma once

#include <memory>
#include <sstream>

using namespace std;

enum class LogLevel {
    ERROR,
    INFO,
    LOW,
    HIGH,
    DEBUG
};

class Logger {
    private:
        static unique_ptr<Logger> instance;
        LogLevel verbosity;

    public:
        Logger(LogLevel defaultVerbosity = LogLevel::INFO);
        ~Logger();

        LogLevel getVerbosity() const;
        void setVerbosity(LogLevel newVerbosity);

        static Logger* getInstance();

        static void print(LogLevel lvl, const string& message);
        static void print(LogLevel lvl, ostringstream& stringStream);
};

#define _LOG(level, message) do { \
    if (level <= Logger::getInstance()->getVerbosity()) { \
        ostringstream _stringStream; \
        _stringStream << "[" << __func__ << ":" << __LINE__ << "] "; \
        _stringStream << message; \
        Logger::print(level, _stringStream); \
    } \
} while(0);

#define LOGE(m) _LOG(LogLevel::ERROR, m)
#define LOGI(m) _LOG(LogLevel::INFO, m)
#define LOGL(m) _LOG(LogLevel::LOW, m)
#define LOGH(m) _LOG(LogLevel::HIGH, m)
#define LOGD(m) _LOG(LogLevel::DEBUG, m)
