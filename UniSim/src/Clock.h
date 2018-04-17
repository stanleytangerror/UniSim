#ifndef CLOCK_H
#define CLOCK_H

#include <ctime>
#include <functional>

class Clock
{
public:
	Clock(bool isPaused = false) 
		: isPaused(isPaused)
		, mRunningFrameCounter(0)
	{ }

	void pause()
	{
		isPaused = true;
	}

	void resume()
	{
		isPaused = false;
	}

	bool paused()
	{
		return isPaused;
	}

	void Tick(float deltaTime)
	{
		if (mRunningFrameCounter > 0)
			mRunningFrameCounter --;
		else
			pause();
	}

	void RunForFrameCount(int frameCount)
	{
		mRunningFrameCounter = frameCount;
		resume();
	}

private:
	std::clock_t * stdClock;
	bool isPaused;
	int mRunningFrameCounter;

public:
	static Clock * Instance()
	{
		if (!Clock::msClock)
			Clock::msClock = new Clock(true);
		
		return Clock::msClock;
	}

private:
	static Clock* msClock;

};

class FrameCounter
{
public:
	using			Functor = std::function<void()>;

					FrameCounter(int frameCount, const Functor& onHit);
	virtual			~FrameCounter() {}

	virtual	void	Tick();

protected:
	Functor	mOnHit;
	int		mFrameCount;
	bool	mAlive;
};

#endif

