#import <Foundation/Foundation.h>
#import <NitroModules/HybridObjectRegistry.hpp>
#import "VisionAuthImpl.hpp"

@interface VisionAuthOnLoad : NSObject
@end

@implementation VisionAuthOnLoad

+ (void)load {
  margelo::nitro::HybridObjectRegistry::registerHybridObjectConstructor(
    "VisionAuth",
    []() -> std::shared_ptr<margelo::nitro::HybridObject> {
      return std::make_shared<margelo::nitro::visionauth::VisionAuthImpl>();
    }
  );
}

@end
