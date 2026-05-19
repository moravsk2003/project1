package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Low-level Gemini HTTP adapter.
 */
public final class GeminiModelClient {
    private final HttpClient httpClient;
    private final String apiKey;
    private final String baseUrl;
    private final Duration timeout;
    private final GeminiResponseParser responseParser;

    public GeminiModelClient(HttpClient httpClient,
                             String apiKey,
                             String baseUrl,
                             Duration timeout,
                             GeminiResponseParser responseParser) {
        this.timeout = timeout != null ? timeout : Duration.ofSeconds(30);
        this.httpClient = httpClient != null
            ? httpClient
            : HttpClient.newBuilder().connectTimeout(this.timeout).build();
        this.apiKey = apiKey != null ? apiKey : "";
        this.baseUrl = baseUrl != null ? baseUrl.replaceAll("/+$", "") : "";
        this.responseParser = responseParser != null ? responseParser : new GeminiResponseParser();
    }

    GeminiModelResult callModel(String modelName, String requestBody) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/models/" + modelName + ":generateContent"))
                .timeout(timeout)
                .header("x-goog-api-key", apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                return GeminiModelResult.error(modelName, modelName + " request failed with HTTP " + response.statusCode());
            }
            String text = responseParser.extractText(response.body());
            String json = responseParser.parseModelJson(text);
            return json != null
                ? GeminiModelResult.decision(modelName, json, text)
                : GeminiModelResult.error(modelName, modelName + " returned non-JSON content");
        } catch (Exception e) {
            return GeminiModelResult.error(modelName, modelName + " request failed: " + e.getMessage());
        }
    }
}
