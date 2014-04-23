// condition_variable example
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

using namespace std;
mutex fprop_start_mutex_, fprop_finished_mutex_;
condition_variable fprop_finish_cond_var_, fprop_start_cond_var_;
bool ready_for_fprop_ = false, fprop_finish_ = false;

void fprop () {
  unique_lock<mutex> lock1(fprop_start_mutex_);
  while (!ready_for_fprop_) {
    // wait for TrainOneBatch to set this off.
    cout  << "Waiting for signal to start fprop " << endl;
    fprop_start_cond_var_.wait(lock1);
    cout  << "Got signal" << endl;
  }
  ready_for_fprop_ = false;
  cout << "Fproping" << endl;
  
  unique_lock<mutex> lock2(fprop_finished_mutex_);
  fprop_finish_ = true;
  fprop_finish_cond_var_.notify_all();
}

void go() {
  cout << "Starting fprop" << endl;
  unique_lock<mutex> lock1(fprop_start_mutex_);
  ready_for_fprop_ = true;
  fprop_start_cond_var_.notify_all();
  cout << "Started" << endl;
  lock1.unlock();

  unique_lock<mutex> lock2(fprop_finished_mutex_);
  while (!fprop_finish_) {
    cout << "Waiting to finish" << endl;
    fprop_finish_cond_var_.wait(lock2);
  }
  fprop_finish_ = false;
  cout << "Done" << endl;

}

int main (int argc, char** argv)
{
  // spawn 10 threads:
  thread th(fprop);
  go();                       // go!

  th.join();

  return 0;
}
