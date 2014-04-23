#ifndef DEFAULT_DATAHANDLER_H_
#define DEFAULT_DATAHANDLER_H_
#include "datahandler.h"

// Looks for [train|val|test]_[data|labels].h5

class DefaultDataHandler : public DataHandler {
 public:
  DefaultDataHandler(const config::DatasetConfig& config);
  void GetBatch(vector<Layer*>& data_layers);
};
#endif
