#include "common/include/threading/SharedMidpointQueue.hpp"
#include <stdexcept>

using namespace std;

bool SharedMidpointQueue::isEmpty() {
    lock_guard<mutex> autoLock(queueMutex);

    return sharedMidpointQueue.empty();
};


float SharedMidpointQueue::dequeueMidpoint() {
    lock_guard<mutex> autoLock(queueMutex);

    if (sharedMidpointQueue.empty()) {
        throw length_error("Queue is empty.");
    }

    float ret = sharedMidpointQueue.front();
    sharedMidpointQueue.pop();

    return ret;
}

void  SharedMidpointQueue::enqueueMidpoint(float m0) {
    lock_guard<mutex> autoLock(queueMutex);
    sharedMidpointQueue.push(m0);
}
