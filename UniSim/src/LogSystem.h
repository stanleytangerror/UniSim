#pragma once

#include <string>

class LogSystem
{
private:
	LogSystem();

public:
	void Log(const std::string & line);

	// singleton impl
public:
	static LogSystem * Instance()
	{
		if (!msInstance)
			msInstance = new LogSystem;

		return msInstance;
	}

private:
	static LogSystem * msInstance;
};