import React, {useState, useRef} from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import {FilePicker} from './FilePicker';
import {type DocumentPickerResponse} from '@react-native-documents/picker';

interface ChatInputProps {
  onSend: (message: string, attachmentIds?: string[]) => void;
  onFilesSelected?: (files: DocumentPickerResponse[]) => void;
  selectedFiles?: DocumentPickerResponse[];
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onFilesSelected,
  selectedFiles = [],
  isLoading = false,
  disabled = false,
  placeholder = 'Ask AmaniQuery...',
}) => {
  const [input, setInput] = useState('');
  const [inputHeight, setInputHeight] = useState(50);
  const inputRef = useRef<TextInput>(null);

  const handleSend = () => {
    if ((input.trim() || selectedFiles.length > 0) && !isLoading && !disabled) {
      onSend(input.trim() || `Uploaded ${selectedFiles.length} file(s)`, []);
      setInput('');
      setInputHeight(50);
      onFilesSelected?.([]);
      Keyboard.dismiss();
    }
  };

  const handleContentSizeChange = (event: any) => {
    const height = Math.min(
      Math.max(50, event.nativeEvent.contentSize.height + 24),
      150,
    );
    setInputHeight(height);
  };

  return (
    <View style={styles.container}>
      {selectedFiles.length > 0 && (
        <View style={styles.filePickerContainer}>
          <FilePicker
            files={selectedFiles}
            onFilesChange={onFilesSelected || (() => {})}
            maxFiles={5}
            maxSizeMB={10}
          />
        </View>
      )}
      
      <View style={styles.inputContainer}>
        {/* Attachment Button */}
        <TouchableOpacity
          style={styles.iconButton}
          onPress={() => {
            // Logic to trigger file picker would go here via parent prop or ref
          }}
          disabled={isLoading || disabled}
          activeOpacity={0.7}>
          <View style={styles.plusIconWrapper}>
            <Icon name="add" size={24} color="#444746" />
          </View>
        </TouchableOpacity>

        {/* Text Input */}
        <TextInput
          ref={inputRef}
          style={[styles.input, {height: inputHeight}]}
          value={input}
          onChangeText={setInput}
          onContentSizeChange={handleContentSizeChange}
          placeholder={placeholder}
          placeholderTextColor="#444746"
          multiline
          maxLength={2000}
          editable={!isLoading && !disabled}
          textAlignVertical="center"
        />

        {/* Right Actions */}
        <View style={styles.rightActions}>
          {!input.trim() && (
            <TouchableOpacity 
              style={styles.iconButton} 
              disabled={isLoading || disabled}>
              <Icon name="mic-outline" size={24} color="#444746" />
            </TouchableOpacity>
          )}
          
          {(input.trim() || selectedFiles.length > 0) && (
            <TouchableOpacity
              style={[styles.sendButton, isLoading && styles.sendButtonDisabled]}
              onPress={handleSend}
              disabled={isLoading || disabled}
              activeOpacity={0.8}>
              {isLoading ? (
                <ActivityIndicator size="small" color="#FFFFFF" />
              ) : (
                <Icon name="send" size={18} color="#FFFFFF" />
              )}
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF', // Or #F0F4F9 for specific gemini bg
  },
  filePickerContainer: {
    marginBottom: 8,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#F0F4F9', // Light gray background like Gemini
    borderRadius: 28,
    paddingHorizontal: 8,
    paddingVertical: 4,
    minHeight: 56,
  },
  iconButton: {
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 2, // Align with bottom
  },
  plusIconWrapper: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#E1E3E1', // Slightly darker circle
    justifyContent: 'center',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: '#1F1F1F',
    marginHorizontal: 4,
    paddingTop: 14,
    paddingBottom: 14,
    // Android padding fix
    paddingVertical: 0, 
  },
  rightActions: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 2,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#004A77', // Gemini Blue/Dark Blue
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 4,
    marginBottom: 4,
  },
  sendButtonDisabled: {
    backgroundColor: '#B0B0B0',
  },
});