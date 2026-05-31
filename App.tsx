import React from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { multiply } from 'react-native-vision-auth';

function App(): React.JSX.Element {
  const result = multiply(3, 7);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.text}>VisionAuth Pure C++ Test</Text>
        <Text style={styles.result}>3 x 7 = {result}</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1E1E1E',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    color: '#00FF00',
    fontWeight: 'bold',
    marginBottom: 20,
  },
  result: {
    fontSize: 32,
    color: '#FFFFFF',
  }
});

export default App;
