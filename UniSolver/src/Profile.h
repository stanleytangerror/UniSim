#pragma once

#include <functional>

class ScopedProfiler
{
public:
	using TimeCount = long long;
	using Functor = std::function<void(const TimeCount, const TimeCount)>;

				ScopedProfiler(const Functor & onProfiled);
	virtual		~ScopedProfiler();

protected:
	TimeCount	mStart;
	Functor		mOnProfiled;
};
