"use client";

import { useState } from "react";
import { Search, Info, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { searchTweets, TweetResult } from "@/lib/api";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TweetResult[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    const data = await searchTweets(query);
    setResults(data);
    setLoading(false);
  };

  return (
    <div className="min-h-screen selection:bg-white selection:text-black text-white">
      {/* Header Navigation */}
      <nav className="fixed top-0 w-full border-b border-white/10 bg-black/50 backdrop-blur-md z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between font-mono text-[10px] tracking-[0.2em]">
          <span className="font-bold uppercase">Abdul / Semantic Search V1.0</span>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto pt-32 pb-20 px-6">
        {/* Hero */}
        <header className="mb-12">
          <h1 className="text-5xl font-black tracking-tighter mb-4 italic uppercase">Tweet Explorer</h1>
          <p className="text-gray-400 font-medium">Latent topical similarity engine.</p>
        </header>

        {/* Search Input */}
        <form onSubmit={handleSearch} className="relative mb-16">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={20} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by intent (e.g., startup growth or discipline)..."
            className="w-full h-14 pl-12 pr-4 bg-transparent border border-white/10 rounded-xl focus:ring-1 focus:ring-white outline-none transition-all font-medium placeholder:text-gray-600"
          />
          <button 
            type="submit" 
            disabled={loading}
            className="absolute right-2 top-2 bottom-2 px-6 bg-white text-black rounded-lg font-bold text-[10px] tracking-widest uppercase hover:opacity-80 transition-opacity disabled:opacity-30"
          >
            {loading ? "Busy..." : "Execute"}
          </button>
        </form>

        {/* Results Area */}
        <div className="space-y-6 mb-24">
          <AnimatePresence mode="popLayout">
            {results.map((res, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-8 border border-white/5 bg-white/5 rounded-2xl"
              >
                <div className="text-[9px] font-mono text-gray-500 mb-4 tracking-[0.3em] uppercase">{res.date}</div>
                <p className="text-xl leading-relaxed font-light text-gray-200">{res.text}</p>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {!loading && results.length === 0 && query && (
             <p className="text-center text-gray-500 text-sm italic">No relevant matches found.</p>
          )}
        </div>

        {/* How It Works Section */}
        <section className="pt-12 border-t border-white/10 text-sm text-gray-400">
          <div className="flex items-center gap-2 mb-6 font-bold text-white uppercase tracking-widest text-[10px]">
            <Info size={14} /> How It Works
          </div>
          
          <div className="space-y-4 leading-relaxed">
            <p>
              This search engine doesn't just look for exact keyword matches. It understands <span className="text-white font-medium italic">meaning and intent</span>.
            </p>
            <p>
              Each tweet in my archive was converted into a numerical representation called an <strong>embedding</strong>. 
              Embeddings capture semantic meaning so similar ideas are placed close together in vector space.
            </p>

            <div className="bg-white/5 p-6 rounded-xl space-y-3 border border-white/5">
              <p className="font-bold text-white text-xs uppercase tracking-tighter">Process Flow:</p>
              <ol className="list-decimal list-inside space-y-2 text-xs font-mono">
                <li>Your search phrase is converted into an embedding.</li>
                <li>The system compares it to all tweet embeddings.</li>
                <li>It finds closest matches using <strong>cosine similarity</strong>.</li>
                <li>Relevant tweets are returned, even without exact word matches.</li>
              </ol>
            </div>
            
            <p className="pt-4 italic">
              Search by <strong>ideas, themes, and intent</strong>, not just keywords.
            </p>
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-20 flex justify-between items-center opacity-40 text-[9px] uppercase font-bold tracking-[0.3em]">
          <span>© 2026 Abdulrahmon</span>
          <a href="https://x.com/cgnito" className="flex items-center gap-1 hover:opacity-100 transition-opacity">
            X Account <ArrowRight size={10} />
          </a>
        </footer>
      </main>
    </div>
  );
}