package com.devoxx.genie.chatmodel.cloud.openai;

import com.devoxx.genie.chatmodel.ChatModelFactory;
import com.devoxx.genie.chatmodel.ThinkingSupport;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import org.jetbrains.annotations.NotNull;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OpenAIChatModelFactory implements ChatModelFactory {

    private static final ModelProvider MODEL_PROVIDER = ModelProvider.OpenAI;

    @Override
    public ChatModel createChatModel(@NotNull CustomChatModel customChatModel) {
        Map<String, Object> customParams = getCustomParams();

        return OpenAiChatModel.builder()
                .apiKey(getApiKey(MODEL_PROVIDER))
                .modelName(customChatModel.getModelName())
                .defaultRequestParameters(createChatContextParameters(customChatModel))
                .maxRetries(customChatModel.getMaxRetries())
                .timeout(Duration.ofSeconds(customChatModel.getTimeout()))
                .returnThinking(ThinkingSupport.isEnabled())
                .customParameters(customParams)
                .listeners(getListener())
                .build();
    }

    private static @NotNull Map<String, Object> getCustomParams() {
        Map<String, Object> customParams = new HashMap<>();
        customParams.put("enable_thinking", true);
        return customParams;
    }

    @Override
    public StreamingChatModel createStreamingChatModel(@NotNull CustomChatModel customChatModel) {
        return OpenAiStreamingChatModel.builder()
                .apiKey(getApiKey(MODEL_PROVIDER))
                .defaultRequestParameters(createChatContextParameters(customChatModel))
                .modelName(customChatModel.getModelName())
                .timeout(Duration.ofSeconds(customChatModel.getTimeout()))
                .returnThinking(ThinkingSupport.isEnabled())
                .customParameters(getCustomParams())
                .listeners(getListener())
                .build();
    }

    @Override
    public List<LanguageModel> getModels() {
        return getModels(MODEL_PROVIDER);
    }

    private ChatRequestParameters createChatContextParameters(@NotNull CustomChatModel customChatModel) {
        String modelName = customChatModel.getModelName().toLowerCase();

        if (modelName.startsWith("o1") || modelName.startsWith("o3")) {
            // o1 and o3 models do not support temperature and topP
            return ChatRequestParameters.builder().build();
        } else if (modelName.startsWith("gpt-5.5")) {
            double temperature = customChatModel.getTemperature();
            // GPT-5.5 requires temperature to be 1
            if (temperature==0) {
                temperature = 1d;
            }
            return ChatRequestParameters.builder()
                    .temperature(temperature)
                    .build();
        } else if (modelName.startsWith("gpt-5")) {
            // GPT-5 supports temperature but not topP
            return ChatRequestParameters.builder()
                    .temperature(customChatModel.getTemperature())
                    .build();
        } else {
            return ChatRequestParameters.builder()
                    .temperature(customChatModel.getTemperature())
                    .topP(customChatModel.getTopP())
                    .build();
        }
    }
}
