import React from 'react';
import { SafeAreaView, Text, View } from 'react-native';
// Import the auto-generated test function from your new C++ library
import { multiply } from 'react-native-vision-auth';

function App(): React.JSX.Element {
  // Call the C++ function synchronously
  const result = multiply(3, 7);

  return (
    <SafeAreaView
      style={{ flex: 1, backgroundColor: '#121212', justifyContent: 'center' }}
    >
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 24, color: '#FFFFFF', marginBottom: 10 }}>
          C++ TurboModule Status:
        </Text>

        {/* If this renders "21", the bridge is fully functional! */}
        <Text style={{ fontSize: 48, color: '#00FF00', fontWeight: 'bold' }}>
          3 x 7 = {result}
        </Text>
      </View>
    </SafeAreaView>
  );
}

export default App;
