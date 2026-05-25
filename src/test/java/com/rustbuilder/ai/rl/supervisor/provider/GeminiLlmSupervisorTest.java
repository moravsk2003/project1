package com.rustbuilder.ai.rl.supervisor.provider;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.Authenticator;
import java.net.CookieHandler;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSession;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GeminiLlmSupervisorTest {

    static class MockHttpResponse implements HttpResponse<String> {
        private final int statusCode;
        private final String body;

        MockHttpResponse(int statusCode, String body) {
            this.statusCode = statusCode;
            this.body = body;
        }

        @Override
        public int statusCode() { return statusCode; }
        @Override
        public HttpRequest request() { return null; }
        @Override
        public Optional<HttpResponse<String>> previousResponse() { return Optional.empty(); }
        @Override
        public HttpHeaders headers() { return HttpHeaders.of(Collections.emptyMap(), (a, b) -> true); }
        @Override
        public String body() { return body; }
        @Override
        public Optional<SSLSession> sslSession() { return Optional.empty(); }
        @Override
        public URI uri() { return null; }
        @Override
        public HttpClient.Version version() { return HttpClient.Version.HTTP_2; }
    }

    static class TestHttpClient extends HttpClient {
        private final java.util.Queue<HttpResponse<String>> responses = new java.util.LinkedList<>();

        void addResponse(int statusCode, String body) {
            responses.add(new MockHttpResponse(statusCode, body));
        }

        @Override
        @SuppressWarnings("unchecked")
        public <T> HttpResponse<T> send(HttpRequest request, HttpResponse.BodyHandler<T> responseBodyHandler) throws IOException, InterruptedException {
            HttpResponse<String> resp = responses.poll();
            if (resp == null) {
                return (HttpResponse<T>) new MockHttpResponse(500, "{\"error\": \"no response mock configured\"}");
            }
            return (HttpResponse<T>) resp;
        }

        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(HttpRequest r, HttpResponse.BodyHandler<T> h) { return null; }
        @Override
        public <T> CompletableFuture<HttpResponse<T>> sendAsync(HttpRequest r, HttpResponse.BodyHandler<T> h, HttpResponse.PushPromiseHandler<T> p) { return null; }
        @Override
        public Optional<CookieHandler> cookieHandler() { return Optional.empty(); }
        @Override
        public Optional<Duration> connectTimeout() { return Optional.empty(); }
        @Override
        public HttpClient.Redirect followRedirects() { return HttpClient.Redirect.NEVER; }
        @Override
        public Optional<ProxySelector> proxy() { return Optional.empty(); }
        @Override
        public SSLContext sslContext() { return null; }
        @Override
        public SSLParameters sslParameters() { return null; }
        @Override
        public Optional<Authenticator> authenticator() { return Optional.empty(); }
        @Override
        public HttpClient.Version version() { return HttpClient.Version.HTTP_2; }
        @Override
        public Optional<Executor> executor() { return Optional.empty(); }
    }

    @Test
    void selectDirectionFailurePreservesErrorMessage() throws Exception {
        TestHttpClient client = new TestHttpClient();
        // Return 403 on the first call
        client.addResponse(403, "Forbidden");

        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("test-key");
        GeminiLlmSupervisor supervisor = new GeminiLlmSupervisor(
            "test-key",
            "gemma-4-31b-it",
            "",
            "https://generativelanguage.googleapis.com/v1beta",
            config,
            RLRewardConfig::createDefault,
            client
        );

        SupervisorObservation obs = observation();
        SupervisorDecision decision = supervisor.review(obs);

        assertEquals(com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.KEEP_GOING, decision.getAction());
        assertTrue(decision.getReason().contains("HTTP 403"), "Expected reason to contain 'HTTP 403', but got: " + decision.getReason());
        
        Map<String, Object> diags = supervisor.getLastDiagnostics();
        assertEquals("SELECT_DIRECTION", diags.get("finalStage"));
        assertTrue(String.valueOf(diags.get("error")).contains("HTTP 403"));
    }

    @Test
    void decideActionFailurePreservesErrorMessage() throws Exception {
        TestHttpClient client = new TestHttpClient();
        // Stage 1 (Direction): Return successful JSON selection
        String directionResponseJson = "{\n" +
            "  \"candidates\": [{\n" +
            "    \"content\": {\n" +
            "      \"parts\": [{\n" +
            "        \"text\": \"{\\n  \\\"direction\\\": \\\"REWARD_TUNING\\\",\\n  \\\"reason\\\": \\\"Tune rewards\\\",\\n  \\\"confidence\\\": 0.9\\n}\"\n" +
            "      }]\n" +
            "    }\n" +
            "  }]\n" +
            "}";
        client.addResponse(200, directionResponseJson);
        // Stage 2 (Decision): Return HTTP 500 error
        client.addResponse(500, "Internal Server Error");

        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("test-key");
        GeminiLlmSupervisor supervisor = new GeminiLlmSupervisor(
            "test-key",
            "gemma-4-31b-it",
            "",
            "https://generativelanguage.googleapis.com/v1beta",
            config,
            RLRewardConfig::createDefault,
            client
        );

        SupervisorObservation obs = observation();
        SupervisorDecision decision = supervisor.review(obs);

        assertEquals(com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.KEEP_GOING, decision.getAction());
        assertTrue(decision.getReason().contains("HTTP 500"), "Expected reason to contain 'HTTP 500', but got: " + decision.getReason());

        Map<String, Object> diags = supervisor.getLastDiagnostics();
        assertEquals("DECIDE_ACTION", diags.get("finalStage"));
        assertTrue(String.valueOf(diags.get("error")).contains("HTTP 500"));
    }

    private SupervisorObservation observation() {
        return new SupervisorObservation(
            "candidate", 10, -1.0, 0.1, 0.2, 1.0, -0.5, 2.0, 0.0, 0, 10, 0.99, 0.1, 32, 5, true, 1, 6, true, 1, 6,
            0.1, 0.2, 0.3, 0.4, 0.5, 100, "MAX_STEPS", "step", "final", "2026-05-12T00:00:00Z", "2026-05-12T00:01:00Z",
            "2026-05-12T01:00:00Z", 60_000L, 3_540_000L, true, false, Map.of(), Map.of(), null,
            Map.of("trainingRunning", true), Map.of(), null
        );
    }
}
