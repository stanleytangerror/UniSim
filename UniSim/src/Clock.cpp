#include "Clock.h"

Clock* Clock::msClock = nullptr;

FrameCounter::FrameCounter(int frameCount, const Functor& onHit)
	: mFrameCount(frameCount)
	, mOnHit(onHit)
	, mAlive(true)
{

}

void FrameCounter::Tick()
{
	if (!mAlive) return;

	if (mFrameCount > 0)
	{
		mFrameCount--;
	}
	else
	{
		mAlive = false;
		mOnHit();
	}
}
