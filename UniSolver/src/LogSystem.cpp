#include "LogSystem.h"
#include <fstream>

void LogSystem::Log(const std::string & line)
{
	std::ofstream outfile;

	outfile.open("log/LogFile.txt", std::ios_base::app);
	outfile << line << std::endl;
}

