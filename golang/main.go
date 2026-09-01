package main

import (
	"fmt"
	"log"
	"os"

	"github.com/YYL424/agentic-rag-studio/api"
	"github.com/YYL424/agentic-rag-studio/config"
	"github.com/YYL424/agentic-rag-studio/service"
)

func main() {
	cfg := config.Load()

	vs := service.NewVectorStoreService()

	kg, err := service.NewKnowledgeGraphService(cfg)
	if err != nil {
		log.Printf("Warning: Neo4j connection failed: %v (running without graph)", err)
		kg = nil
	}

	_ = os.MkdirAll("./uploads", 0o755)

	server := api.NewServer(cfg, vs, kg)
	fmt.Printf("Agentic RAG Studio (Go) starting on port %s\n", cfg.APIPort)
	if err := server.Run(); err != nil {
		log.Fatal(err)
	}
}
