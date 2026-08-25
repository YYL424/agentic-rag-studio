package com.agenthub.agent;

import com.agenthub.model.DocumentChunk;
import org.apache.tika.Tika;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * 文档解析 Agent (Java版)
 *
 * 当前原型使用 Apache Tika 提取文本。Python 解析服务的 HTTP 集成仍是后续工作。
 */
@Component
public class DocParserAgent {

    private static final int CHUNK_SIZE = 512;
    private static final int CHUNK_OVERLAP = 64;

    private final Tika tika = new Tika();
    public List<DocumentChunk> parse(String filePath) throws IOException {
        File file = new File(filePath);
        String docId = computeDocId(filePath);
        String docType = detectType(file);

        String rawText = extractText(file);
        return chunkText(rawText, docId, docType, filePath);
    }

    public List<DocumentChunk> parseBatch(List<String> filePaths) throws IOException {
        List<DocumentChunk> allChunks = new ArrayList<>();
        for (String path : filePaths) {
            allChunks.addAll(parse(path));
        }
        return allChunks;
    }

    private String extractText(File file) throws IOException {
        return tika.parseToString(file);
    }

    private String detectType(File file) {
        String name = file.getName().toLowerCase();
        if (name.endsWith(".pdf")) return "pdf";
        if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image";
        if (name.endsWith(".csv") || name.endsWith(".xlsx")) return "table";
        if (name.endsWith(".md")) return "markdown";
        return "text";
    }

    private List<DocumentChunk> chunkText(String text, String docId, String docType, String source) {
        List<DocumentChunk> chunks = new ArrayList<>();
        int idx = 0;
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + CHUNK_SIZE, text.length());
            String content = text.substring(start, end).trim();
            if (!content.isEmpty()) {
                chunks.add(DocumentChunk.builder()
                        .chunkId(docId + "#chunk-" + idx)
                        .docId(docId)
                        .chunkIndex(idx)
                        .content(content)
                        .docType(docType)
                        .metadata(Map.of("source", source, "charStart", start, "charEnd", end))
                        .build());
                idx++;
            }
            if (end == text.length()) {
                break;
            }
            start = end - CHUNK_OVERLAP;
        }
        return chunks;
    }

    private static String computeDocId(String path) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(path.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash).substring(0, 16);
        } catch (Exception e) {
            return String.valueOf(path.hashCode());
        }
    }
}
