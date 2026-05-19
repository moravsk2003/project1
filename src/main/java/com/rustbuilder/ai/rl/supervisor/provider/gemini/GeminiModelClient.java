package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.Set;

/**
 * Low-level Gemini HTTP adapter.
 */
public final class GeminiModelClient {
    private static final int MAX_RETRIES = 3;
    private static final long BASE_BACKOFF_MS = 2000;
    private static final Set<Integer> RETRYABLE_STATUS_CODES = Set.of(500, 502, 503);

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
        this.timeout = timeout != null ? timeout : Duration.ofSeconds(90);
        this.httpClient = httpClient != null
            ? httpClient
            : HttpClient.newBuilder().connectTimeout(this.timeout).build();
        this.apiKey = apiKey != null ? apiKey : "";
        this.baseUrl = baseUrl != null ? baseUrl.replaceAll("/+$", "") : "";
        this.responseParser = responseParser != null ? responseParser : new GeminiResponseParser();
    }

    GeminiModelResult callModel(String modelName, String requestBody) {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/models/" + modelName + ":generateContent"))
            .timeout(timeout)
            .header("x-goog-api-key", apiKey)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        String lastError = null;
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                int status = response.statusCode();

                if (status >= 200 && status < 300) {
                    String text = responseParser.extractText(response.body());
                    String json = responseParser.parseModelJson(text);
                    return json != null
                        ? GeminiModelResult.decision(modelName, json, text)
                        : GeminiModelResult.error(modelName, modelName + " returned non-JSON content");
                }

                lastError = modelName + " request failed with HTTP " + status;
                if (!RETRYABLE_STATUS_CODES.contains(status)) {
                    // Non-retryable HTTP error (e.g. 400, 401, 404) — fail immediately
                    return GeminiModelResult.error(modelName, lastError);
                }

                System.err.printf("[GeminiModelClient] %s attempt %d/%d failed (HTTP %d), retrying in %dms...%n",
                    modelName, attempt, MAX_RETRIES, status, backoffMs(attempt));

            } catch (HttpTimeoutException e) {
                lastError = modelName + " request timed out (attempt " + attempt + "/" + MAX_RETRIES + ")";
                System.err.printf("[GeminiModelClient] %s attempt %d/%d timed out, retrying in %dms...%n",
                    modelName, attempt, MAX_RETRIES, backoffMs(attempt));
            } catch (java.io.IOException e) {
                lastError = modelName + " I/O error: " + e.getMessage();
                System.err.printf("[GeminiModelClient] %s attempt %d/%d I/O error: %s, retrying in %dms...%n",
                    modelName, attempt, MAX_RETRIES, e.getMessage(), backoffMs(attempt));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return GeminiModelResult.error(modelName, modelName + " request interrupted");
            } catch (Exception e) {
                return GeminiModelResult.error(modelName, modelName + " request failed: " + e.getMessage());
            }

            // Backoff before next attempt (unless this was the last attempt)
            if (attempt < MAX_RETRIES) {
                try {
                    Thread.sleep(backoffMs(attempt));
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return GeminiModelResult.error(modelName, modelName + " retry interrupted");
                }
            }
        }

        return GeminiModelResult.error(modelName, lastError + " (after " + MAX_RETRIES + " attempts)");
    }

    private static long backoffMs(int attempt) {
        return BASE_BACKOFF_MS * (1L << (attempt - 1));
    }
}
