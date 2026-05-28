// react-native.config.js
const path = require('path');

module.exports = {
  dependencies: {
    'react-native-vision-auth': {
      // Bypasses the pnpm symlink and points Gradle directly to the physical files
      root: path.resolve(__dirname, './modules/react-native-vision-auth'),
    },
  },
};
