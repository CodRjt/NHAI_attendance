const path = require('path');

module.exports = {
  dependencies: {
    'react-native-vision-auth': {
      // Explicitly point RN CLI to the actual local folder, bypassing the symlink
      root: path.resolve(__dirname, './modules/react-native-vision-auth'),
    },
  },
};
