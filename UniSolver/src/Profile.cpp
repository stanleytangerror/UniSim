#include "Profile.h"
#include <chrono>

namespace
{
	long long GetNanoSecondCount()
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::high_resolution_clock::now().time_since_epoch()
			).count();
	}
}

ScopedProfiler::ScopedProfiler(const Functor & onProfiled)
	: mStart(GetNanoSecondCount())
	, mOnProfiled(onProfiled)
{

}

ScopedProfiler::~ScopedProfiler()
{
	mOnProfiled(mStart, GetNanoSecondCount());
}

