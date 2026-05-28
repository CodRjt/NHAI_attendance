#import <Foundation/Foundation.h>
#import "VisionAuthImpl.h"
#import <ReactCommon/CxxTurboModuleUtils.h>

@interface VisionAuthOnLoad : NSObject
@end

@implementation VisionAuthOnLoad

using namespace facebook::react;

+ (void)load
{
  registerCxxModuleToGlobalModuleMap(
    std::string(VisionAuthImpl::kModuleName),
    [](std::shared_ptr<CallInvoker> jsInvoker) {
      return std::make_shared<VisionAuthImpl>(jsInvoker);
    }
  );
}

@end
