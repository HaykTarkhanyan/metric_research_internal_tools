# Makefile for automated git workflow
# Usage: make push m="Your commit message here"

.PHONY: push help

# Default target
help:
	@echo "Available targets:"
	@echo "  push m=\"message\"  - Pull, add, commit with message, and push"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make push m=\"Added new feature\""

# Main target for git workflow
push:
ifndef m
	$(error m is undefined. Usage: make push m="Your commit message")
endif
	@echo "🔄 Pulling latest changes..."
	git pull
	@echo "📝 Adding all changes..."
	git add .
	@echo "💾 Committing with message: $(m)"
	git commit -m "$(m)"
	@echo "🚀 Pushing to remote..."
	git push
	@echo "✅ Done!"