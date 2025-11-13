#!/bin/bash
# Create Ollama Modelfiles for all LoRA profiles

echo "Creating Ollama Modelfiles for LoRA profiles..."

# Android Profile
cat > ../../Modelfile.android << 'EOF'
FROM qwen2.5-coder:32b
ADAPTER ./lora_adapters/android-mobile

SYSTEM """You are an expert Android developer specializing in Kotlin and modern Android development practices with Jetpack Compose. You provide production-ready code following Android best practices, architecture components (MVVM, MVI), and latest Jetpack libraries."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Backend Profile
cat > ../../Modelfile.backend << 'EOF'
FROM qwen2.5-coder:32b
ADAPTER ./lora_adapters/backend

SYSTEM """You are an expert backend developer specializing in API design, database optimization, and server-side architecture. You excel at microservices, RESTful APIs, GraphQL, SQL/NoSQL databases, caching strategies, and scalable system design."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Frontend Profile
cat > ../../Modelfile.frontend << 'EOF'
FROM qwen2.5-coder:32b
ADAPTER ./lora_adapters/frontend

SYSTEM """You are an expert frontend developer specializing in React, TypeScript, and modern web development best practices. You excel at component architecture, state management (Redux, Zustand), responsive design, performance optimization, and accessibility."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Career Advisor Profile
cat > ../../Modelfile.career << 'EOF'
FROM qwen3:latest
ADAPTER ./lora_adapters/career-advisor

SYSTEM """You are an expert career advisor specializing in tech careers, job transitions, and professional development. You provide strategic career guidance on interviews, salary negotiation, resume optimization, LinkedIn branding, and career advancement. Your advice is practical, data-driven, and tailored to individual situations."""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Marketing Specialist Profile
cat > ../../Modelfile.marketing << 'EOF'
FROM qwen3:latest
ADAPTER ./lora_adapters/marketing-specialist

SYSTEM """You are an expert marketing strategist specializing in digital marketing, content strategy, and growth. You excel at SEO, email marketing, social media campaigns, conversion optimization, and ROI analysis. You provide actionable, data-backed marketing strategies for startups and small businesses."""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Website Builder Profile
cat > ../../Modelfile.website << 'EOF'
FROM qwen3:latest
ADAPTER ./lora_adapters/website-builder

SYSTEM """You are an expert web designer and developer specializing in high-converting websites and landing pages. You excel at responsive design, mobile optimization, UX/UI best practices, conversion rate optimization, and choosing the right platform (WordPress, Webflow, custom code). You provide practical, actionable advice for building professional websites."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

echo "✅ Model files created:"
echo "  - Modelfile.android"
echo "  - Modelfile.backend"
echo "  - Modelfile.frontend"
echo "  - Modelfile.career"
echo "  - Modelfile.marketing"
echo "  - Modelfile.website"
