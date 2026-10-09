import React, {useState} from 'react';
import {
  View,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import {type DocumentPickerResponse} from '@react-native-documents/picker';
import {useChat} from '../../hooks/useChat';
import {chatAPI} from '../../api/chat';
import {MessageList} from './MessageList';
import {ChatInput} from './ChatInput';
import {ErrorMessage} from '../common/ErrorMessage';
import {EmptyState} from '../common/EmptyState';
import {TypingIndicator} from '../common/TypingIndicator';

export const ChatScreen: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<DocumentPickerResponse[]>(
    [],
  );
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const {
    messages,
    currentSessionId,
    isLoading,
    error,
    sendMessage,
    submitFeedback,
    createNewSession,
    editMessage,
    regenerateMessage,
  } = useChat();

  const handleSend = async (message: string, attachmentIds?: string[]) => {
    // If editing, use editMessage instead
    if (editingMessageId) {
      await editMessage(editingMessageId, message);
      setEditingMessageId(null);
      return;
    }

    // Upload files first if any
    let finalAttachmentIds = attachmentIds || [];
    if (selectedFiles.length > 0) {
      try {
        setUploadingFiles(true);
        let sessionId = currentSessionId;
        if (!sessionId) {
          sessionId = await createNewSession(
            message.trim().substring(0, 50) || 'File upload',
          );
          if (!sessionId) {
            Alert.alert('Error', 'Failed to create session');
            setUploadingFiles(false);
            return;
          }
        }

        const uploadPromises = selectedFiles.map(file =>
          chatAPI.uploadAttachment(
            sessionId!,
            file.uri,
            file.name || 'file',
            file.type || 'application/octet-stream',
          ),
        );

        finalAttachmentIds = await Promise.all(uploadPromises);
        setSelectedFiles([]);
      } catch (err) {
        Alert.alert(
          'Upload Error',
          err instanceof Error ? err.message : 'Failed to upload files',
        );
        setUploadingFiles(false);
        return;
      } finally {
        setUploadingFiles(false);
      }
    }

    // Send message with attachment IDs
    await sendMessage(message, true, finalAttachmentIds);
  };

  const handleEdit = (messageId: string) => {
    const message = messages.find(m => m.id === messageId);
    if (message && message.role === 'user') {
      setEditingMessageId(messageId);
      // TODO: Pre-fill input with message content
      // This would require passing the input value up or using a ref
    }
  };

  const handleRegenerate = async (messageId: string) => {
    await regenerateMessage(messageId);
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}>
      <View style={styles.content}>
        {error && (
          <ErrorMessage
            message={error}
            onDismiss={() => {
              // Error dismissal handled by hook
            }}
          />
        )}

        {messages.length === 0 && !isLoading ? (
          <EmptyState />
        ) : (
          <>
            <MessageList
              messages={messages}
              onFeedback={submitFeedback}
              onEdit={handleEdit}
              onRegenerate={handleRegenerate}
            />
            {isLoading && <TypingIndicator />}
          </>
        )}
      </View>

      <ChatInput
        onSend={handleSend}
        onFilesSelected={setSelectedFiles}
        selectedFiles={selectedFiles}
        isLoading={isLoading || uploadingFiles}
        placeholder={
          editingMessageId
            ? 'Edit your message...'
            : 'Ask about Kenyan law, parliament, or news...'
        }
      />
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    flex: 1,
  },
});
