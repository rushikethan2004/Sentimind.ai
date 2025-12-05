import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Smile, Frown, Meh, Send, Trash2, Activity, MessageSquare, 
  TrendingUp, AlertCircle, Upload, FileText, Bot, X, Zap,
  Sparkles, MessageCircle, Copy, Check, RefreshCw, ShieldCheck, 
  FileSpreadsheet, FileType, LayoutDashboard, ListFilter
} from 'lucide-react';

/**
 * --- GEMINI API UTILITIES ---
 */
const callGemini = async (prompt, systemInstruction = "") => {
  const apiKey = ""; // Injected at runtime
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
  
  const payload = {
    contents: [{ parts: [{ text: prompt }] }],
    systemInstruction: { parts: [{ text: systemInstruction }] }
  };

  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  
  for (let i = 0; i < 3; i++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";
    } catch (error) {
      if (i === 2) return "Error generating AI response. Please try again.";
      await delay(1000 * Math.pow(2, i)); // Exponential backoff
    }
  }
};

/**
 * --- MACHINE LEARNING ENGINE: NAIVE BAYES CLASSIFIER ---
 * Client-side training and prediction.
 */
class NaiveBayesClassifier {
  constructor() {
    this.wordCounts = { Positive: {}, Negative: {}, Neutral: {} };
    this.classCounts = { Positive: 0, Negative: 0, Neutral: 0 };
    this.vocab = new Set();
    this.totalDocs = 0;
  }

  tokenize(text) {
    if (!text) return [];
    return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 2);
  }

  train(documents) {
    this.wordCounts = { Positive: {}, Negative: {}, Neutral: {} };
    this.classCounts = { Positive: 0, Negative: 0, Neutral: 0 };
    this.vocab = new Set();
    this.totalDocs = 0;

    documents.forEach(doc => {
      this.totalDocs++;
      const category = doc.label; 
      this.classCounts[category]++;
      
      const tokens = this.tokenize(doc.text);
      tokens.forEach(token => {
        this.vocab.add(token);
        this.wordCounts[category][token] = (this.wordCounts[category][token] || 0) + 1;
      });
    });
  }

  predict(text) {
    const tokens = this.tokenize(text);
    const categories = ['Positive', 'Negative', 'Neutral'];
    let bestCategory = 'Neutral';
    let maxProb = -Infinity;
    let scores = {};

    categories.forEach(category => {
      let logProb = Math.log((this.classCounts[category] || 0.1) / (this.totalDocs || 1));
      tokens.forEach(token => {
        const tokenCount = this.wordCounts[category][token] || 0;
        const classTotalWords = Object.values(this.wordCounts[category]).reduce((a, b) => a + b, 0);
        const vocabSize = this.vocab.size;
        logProb += Math.log((tokenCount + 1) / (classTotalWords + vocabSize));
      });
      scores[category] = logProb;
      if (logProb > maxProb) {
        maxProb = logProb;
        bestCategory = category;
      }
    });
    return { label: bestCategory, scores };
  }
}

const classifier = new NaiveBayesClassifier();
const STOPWORDS = new Set(['the', 'and', 'but', 'for', 'with', 'was', 'that', 'this', 'have', 'are', 'not', 'can', 'you', 'your', 'product', 'app', 'service']);

/**
 * --- REACT COMPONENTS ---
 */

const ChatBot = ({ history, onClose }) => {
  const [messages, setMessages] = useState([
    { role: 'bot', text: "Hi! I'm your AI Analyst. I have access to all your current reviews. Ask me anything!" }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // Prepare context for Gemini
    const dataContext = JSON.stringify(history.slice(0, 50)); 
    const systemPrompt = `You are a helpful Data Analyst assistant. You are analyzing a dataset of customer reviews. 
    Here is the data in JSON format: ${dataContext}. 
    Answer the user's questions based strictly on this data. If the answer isn't in the data, say so. 
    Keep answers concise and professional.`;

    const response = await callGemini(input, systemPrompt);
    
    setIsTyping(false);
    setMessages(prev => [...prev, { role: 'bot', text: response }]);
  };

  return (
    <div className="fixed bottom-4 right-4 w-96 bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden z-50 h-[500px]">
      <div className="bg-indigo-600 p-4 flex justify-between items-center text-white">
        <div className="flex items-center gap-2">
          <Sparkles size={18} className="text-yellow-300" />
          <span className="font-semibold text-sm">Gemini Analyst</span>
        </div>
        <button onClick={onClose} className="hover:bg-indigo-700 p-1 rounded"><X size={16} /></button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 bg-slate-50 space-y-4" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-lg text-sm ${m.role === 'user' ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-white border border-slate-200 text-slate-700 rounded-bl-none shadow-sm'}`}>
              {m.text}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
             <div className="bg-white border border-slate-200 px-4 py-3 rounded-xl rounded-bl-none shadow-sm flex gap-1">
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-75"></span>
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-150"></span>
             </div>
          </div>
        )}
      </div>
      <div className="p-3 bg-white border-t border-slate-100 flex gap-2">
        <input 
          className="flex-1 bg-slate-100 border-0 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
          placeholder="Ask about trends, issues..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button onClick={handleSend} disabled={isTyping} className="bg-indigo-600 text-white p-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
          <Send size={18} />
        </button>
      </div>
    </div>
  );
};

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'results'
  const [input, setInput] = useState('');
  const [showChat, setShowChat] = useState(false);
  const [history, setHistory] = useState([
    { id: 1, text: "The product is great, fast delivery!", label: "Positive", timestamp: "Initial Data" },
    { id: 2, text: "Terrible service, very slow and rude.", label: "Negative", timestamp: "Initial Data" },
    { id: 3, text: "It's okay, nothing special.", label: "Neutral", timestamp: "Initial Data" },
    { id: 4, text: "I love the new features.", label: "Positive", timestamp: "Initial Data" },
    { id: 5, text: "Broken immediately. Waste of money.", label: "Negative", timestamp: "Initial Data" }
  ]);
  
  const [prediction, setPrediction] = useState({ label: 'Neutral', scores: {} });
  
  // AI Feature States
  const [aiReport, setAiReport] = useState(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [draftingReplyId, setDraftingReplyId] = useState(null);
  const [generatedReplies, setGeneratedReplies] = useState({}); 
  
  // New AI Feature States
  const [isGeneratingData, setIsGeneratingData] = useState(false);
  const [aiVerification, setAiVerification] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [libsLoaded, setLibsLoaded] = useState(false);
  const [isProcessingFile, setIsProcessingFile] = useState(false);

  // Load Parsing Libraries Dynamically
  useEffect(() => {
    const loadScript = (src) => {
      return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve();
          return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
      });
    };

    Promise.all([
      loadScript('https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js'), // SheetJS for Excel
      loadScript('https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.6.0/mammoth.browser.min.js') // Mammoth for Word
    ]).then(() => {
      setLibsLoaded(true);
      console.log('File parsing libraries loaded');
    }).catch(err => console.error('Failed to load libraries', err));
  }, []);

  // ML Training Trigger
  useEffect(() => {
    classifier.train(history);
    if (input) setPrediction(classifier.predict(input));
  }, [history]); 

  // Real-time prediction
  useEffect(() => {
    if (!input.trim()) {
      setPrediction({ label: 'Neutral', scores: {} });
      setAiVerification(null); 
      return;
    }
    setPrediction(classifier.predict(input));
    setAiVerification(null); 
  }, [input]);

  const handleSave = () => {
    if (!input.trim()) return;
    const newItem = {
      id: Date.now(),
      text: input,
      label: prediction.label,
      timestamp: new Date().toLocaleTimeString()
    };
    setHistory(prev => [newItem, ...prev]);
    setInput('');
  };

  const handleCorrection = (id, newLabel) => {
    setHistory(prev => prev.map(item => item.id === id ? { ...item, label: newLabel } : item));
  };

  // --- ENHANCED FILE UPLOAD HANDLER ---
  const processExtractedText = (text) => {
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 5);
    const newEntries = lines.map((line, idx) => ({
      id: Date.now() + idx,
      text: line.trim(),
      label: classifier.predict(line).label,
      timestamp: "Imported File"
    }));
    setHistory(prev => [...newEntries, ...prev]);
    setIsProcessingFile(false);
    setActiveTab('results'); // Switch to results tab after upload
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setIsProcessingFile(true);
    const fileName = file.name.toLowerCase();

    // 1. CSV Handler
    if (fileName.endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = (event) => processExtractedText(event.target.result);
      reader.readAsText(file);
    }
    // 2. Excel Handler (.xlsx, .xls)
    else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
      if (!window.XLSX) {
        alert("Excel parser is still loading. Please try again in a moment.");
        setIsProcessingFile(false);
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = window.XLSX.read(data, { type: 'array' });
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        const jsonData = window.XLSX.utils.sheet_to_json(firstSheet, { header: 1 });
        const text = jsonData.flat().join('\n');
        processExtractedText(text);
      };
      reader.readAsArrayBuffer(file);
    }
    // 3. Word Handler (.docx)
    else if (fileName.endsWith('.docx')) {
      if (!window.mammoth) {
        alert("Word parser is still loading. Please try again in a moment.");
        setIsProcessingFile(false);
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const arrayBuffer = e.target.result;
        window.mammoth.extractRawText({ arrayBuffer: arrayBuffer })
          .then(result => processExtractedText(result.value))
          .catch(err => {
            console.error(err);
            alert("Error parsing Word document.");
            setIsProcessingFile(false);
          });
      };
      reader.readAsArrayBuffer(file);
    }
    // 4. Text Fallback
    else {
      const reader = new FileReader();
      reader.onload = (event) => processExtractedText(event.target.result);
      reader.readAsText(file);
    }
  };

  // --- GEMINI FUNCTIONS ---
  const generateInsightReport = async () => {
    setIsGeneratingReport(true);
    const dataContext = JSON.stringify(history.slice(0, 30)); 
    const prompt = `Analyze these customer reviews and generate a concise insight report. 
    1. Overall Sentiment Trend.
    2. Top 3 Specific Complaints (if any).
    3. Top 3 Praised Features (if any).
    4. One Actionable Recommendation for the business.
    Reviews: ${dataContext}`;
    const report = await callGemini(prompt, "You are an expert business consultant.");
    setAiReport(report);
    setIsGeneratingReport(false);
  };

  const generateSmartReply = async (item) => {
    setDraftingReplyId(item.id);
    const prompt = `Write a polite, professional, and concise customer service response to this review. 
    Review: "${item.text}"
    Sentiment: ${item.label}
    The response should address their specific point. If negative, apologize and offer help. If positive, thank them warmly.`;
    const reply = await callGemini(prompt);
    setGeneratedReplies(prev => ({ ...prev, [item.id]: reply }));
    setDraftingReplyId(null);
  };

  const generateSyntheticData = async () => {
    setIsGeneratingData(true);
    const prompt = `Generate 5 diverse, realistic customer reviews for a SaaS or E-commerce product. 
    Include 2 positive, 2 negative, and 1 neutral review.
    Format ONLY as a valid JSON array of objects with keys: "text" and "label".
    Example: [{"text": "Love it", "label": "Positive"}]`;
    const response = await callGemini(prompt, "You are a data generator. Output only JSON.");
    try {
        const jsonStr = response.replace(/```json|```/g, '').trim();
        const data = JSON.parse(jsonStr);
        if (Array.isArray(data)) {
            const newItems = data.map((item, i) => ({
                id: Date.now() + i,
                text: item.text,
                label: item.label,
                timestamp: "AI Generated"
            }));
            setHistory(prev => [...newItems, ...prev]);
            setActiveTab('results'); // Switch to results to show data
        }
    } catch (e) { console.error("Failed to parse AI data", e); }
    setIsGeneratingData(false);
  };

  const verifySentiment = async () => {
    if (!input) return;
    setIsVerifying(true);
    const prompt = `Analyze the sentiment of this text deeply. Detect sarcasm, nuance, or mixed feelings.
    Text: "${input}"
    Output strictly in this format:
    Sentiment: [Positive/Negative/Neutral]
    Confidence: [High/Medium/Low]
    Reasoning: [One sentence explanation]`;
    const response = await callGemini(prompt, "You are a sentiment expert.");
    setAiVerification(response);
    setIsVerifying(false);
  };

  const copyReply = (text) => navigator.clipboard.writeText(text);

  const stats = {
    pos: history.filter(h => h.label === 'Positive').length,
    neg: history.filter(h => h.label === 'Negative').length,
    neu: history.filter(h => h.label === 'Neutral').length
  };

  const getLabelColor = (l) => {
    if (l === 'Positive') return 'text-emerald-600 bg-emerald-50 border-emerald-200';
    if (l === 'Negative') return 'text-rose-600 bg-rose-50 border-rose-200';
    return 'text-slate-600 bg-slate-50 border-slate-200';
  };

  // Group history for the Results Page
  const categorizedHistory = useMemo(() => {
    return {
      Positive: history.filter(h => h.label === 'Positive'),
      Neutral: history.filter(h => h.label === 'Neutral'),
      Negative: history.filter(h => h.label === 'Negative')
    };
  }, [history]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="text-indigo-600" />
            <h1 className="font-bold text-xl tracking-tight text-slate-800">
              Sentimind<span className="text-indigo-600">.ai</span>
            </h1>
          </div>
          
          {/* Main Navigation Tabs */}
          <div className="flex bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'dashboard' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              <LayoutDashboard size={16} />
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('results')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'results' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              <ListFilter size={16} />
              Detailed Results
            </button>
          </div>

          <div className="flex items-center gap-3">
             <button 
              onClick={() => setShowChat(!showChat)}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-lg hover:shadow-md transition-all font-medium text-sm"
             >
               <Sparkles size={16} className="text-yellow-300" />
               Ask AI Analyst
             </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {activeTab === 'dashboard' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in zoom-in-95 duration-300">
            {/* LEFT COLUMN: Input & Upload */}
            <div className="lg:col-span-7 space-y-6">
              
              {/* Main Input Area */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                  <label className="text-sm font-semibold text-slate-600 flex items-center gap-2">
                    <MessageSquare size={16} />
                    Analyze Feedback
                  </label>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-slate-400">Model: Naive Bayes v1.0</span>
                  </div>
                </div>
                <div className="p-4">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type a review to classify it (e.g., 'The shipping was super fast')..."
                    className="w-full h-32 p-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all resize-none text-slate-700 text-lg"
                  />
                  
                  <div className="flex justify-between items-center mt-4">
                    <div className="flex items-center gap-2">
                        {/* Prediction Badge */}
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${getLabelColor(prediction.label)}`}>
                        {prediction.label === 'Positive' && <Smile size={18} />}
                        {prediction.label === 'Negative' && <Frown size={18} />}
                        {prediction.label === 'Neutral' && <Meh size={18} />}
                        <span className="font-bold text-sm">{prediction.label}</span>
                        </div>

                        {/* AI Verification Button */}
                        {input.trim() && (
                            <button 
                                onClick={verifySentiment} 
                                disabled={isVerifying}
                                className="text-xs flex items-center gap-1 text-indigo-600 hover:text-indigo-800 font-medium px-2 py-1 rounded hover:bg-indigo-50 transition-colors"
                            >
                                {isVerifying ? <RefreshCw size={14} className="animate-spin"/> : <ShieldCheck size={14} />}
                                {isVerifying ? "Checking..." : "Verify with AI"}
                            </button>
                        )}
                    </div>

                    <button
                      onClick={handleSave}
                      disabled={!input.trim()}
                      className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 shadow-sm"
                    >
                      <Send size={16} />
                      Add to Training Data
                    </button>
                  </div>

                  {/* AI Verification Result Box */}
                  {aiVerification && (
                    <div className="mt-3 p-3 bg-indigo-50 border border-indigo-100 rounded-lg text-sm text-indigo-900 flex gap-3 items-start animate-in fade-in slide-in-from-top-2">
                        <Sparkles size={16} className="text-indigo-500 mt-0.5 flex-shrink-0" />
                        <div className="whitespace-pre-wrap leading-relaxed">
                            {aiVerification}
                        </div>
                    </div>
                  )}
                </div>
              </div>

              {/* File Upload Section */}
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                 <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                        <Upload size={18} className="text-indigo-500"/> 
                        Batch Analysis
                      </h3>
                      <div className="flex flex-wrap gap-2 mt-2">
                        <span className="text-[10px] font-medium bg-blue-50 text-blue-600 px-2 py-1 rounded border border-blue-100 flex items-center gap-1">
                          <FileType size={12}/> .TXT
                        </span>
                        <span className="text-[10px] font-medium bg-green-50 text-green-600 px-2 py-1 rounded border border-green-100 flex items-center gap-1">
                          <FileSpreadsheet size={12}/> .CSV
                        </span>
                        <span className="text-[10px] font-medium bg-emerald-50 text-emerald-600 px-2 py-1 rounded border border-emerald-100 flex items-center gap-1">
                          <FileSpreadsheet size={12}/> .XLSX
                        </span>
                        <span className="text-[10px] font-medium bg-indigo-50 text-indigo-600 px-2 py-1 rounded border border-indigo-100 flex items-center gap-1">
                          <FileText size={12}/> .DOCX
                        </span>
                      </div>
                    </div>
                 </div>
                 <label className={`flex flex-col items-center justify-center w-full h-24 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${isProcessingFile ? 'bg-slate-100 border-slate-300' : 'bg-slate-50 border-slate-300 hover:bg-slate-100'}`}>
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        {isProcessingFile ? (
                          <div className="flex items-center gap-2 text-indigo-600 font-medium">
                            <RefreshCw size={18} className="animate-spin" />
                            Processing File...
                          </div>
                        ) : (
                          <p className="text-sm text-slate-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                        )}
                        <p className="text-xs text-slate-400 mt-1">Supports TXT, CSV, Excel, Word</p>
                    </div>
                    <input 
                      type="file" 
                      className="hidden" 
                      accept=".txt,.csv,.xlsx,.xls,.docx" 
                      onChange={handleFileUpload} 
                      disabled={isProcessingFile}
                    />
                </label>
              </div>
            </div>

            {/* RIGHT COLUMN: Stats & AI Report (Moved here) */}
            <div className="lg:col-span-5 space-y-6">
              
              {/* Stats Cards */}
              <div className="grid grid-cols-3 gap-3">
                 <div className="bg-emerald-50 p-3 rounded-xl border border-emerald-100 text-center">
                    <div className="text-2xl font-bold text-emerald-700">{stats.pos}</div>
                    <div className="text-xs font-semibold text-emerald-600 uppercase">Positive</div>
                 </div>
                 <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                    <div className="text-2xl font-bold text-slate-700">{stats.neu}</div>
                    <div className="text-xs font-semibold text-slate-500 uppercase">Neutral</div>
                 </div>
                 <div className="bg-rose-50 p-3 rounded-xl border border-rose-100 text-center">
                    <div className="text-2xl font-bold text-rose-700">{stats.neg}</div>
                    <div className="text-xs font-semibold text-rose-600 uppercase">Negative</div>
                 </div>
              </div>

              {/* Synthetic Data Generator (Tool Card) */}
              <div className="bg-indigo-50 rounded-xl p-4 border border-indigo-100 flex items-center justify-between">
                <div>
                   <h4 className="font-semibold text-indigo-900 text-sm">Need Training Data?</h4>
                   <p className="text-xs text-indigo-700/80">Generate fake reviews to test the system.</p>
                </div>
                <button 
                  onClick={generateSyntheticData}
                  disabled={isGeneratingData}
                  className="text-xs font-medium text-white bg-indigo-600 px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-1 disabled:opacity-50 shadow-sm"
                >
                  {isGeneratingData ? <RefreshCw size={12} className="animate-spin"/> : <Sparkles size={12}/>}
                  {isGeneratingData ? "Creating..." : "Generate"}
                </button>
              </div>

              {/* AI Insights Section (Moved from left) */}
              <div className="bg-gradient-to-br from-indigo-50 to-white rounded-xl p-6 border border-indigo-100 shadow-sm h-[400px] flex flex-col">
                 <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                      <Sparkles size={18} className="text-indigo-500"/> 
                      AI Deep Insights
                    </h3>
                    <button 
                      onClick={generateInsightReport}
                      disabled={isGeneratingReport}
                      className="text-xs font-medium bg-white border border-indigo-200 text-indigo-700 px-3 py-1.5 rounded-full hover:bg-indigo-50 transition-colors flex items-center gap-1 disabled:opacity-50"
                    >
                      {isGeneratingReport ? 'Analyzing...' : 'Generate Report ✨'}
                    </button>
                 </div>
                 
                 <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {aiReport ? (
                      <div className="prose prose-sm max-w-none text-slate-600 bg-white/50 p-4 rounded-lg border border-indigo-50">
                        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">{aiReport}</pre>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-slate-400 text-sm border-2 border-dashed border-indigo-100 rounded-lg h-full flex items-center justify-center">
                        <p className="px-8">Click generate above to get a comprehensive analysis of your {history.length} reviews using Gemini.</p>
                      </div>
                    )}
                 </div>
              </div>

            </div>
          </div>
        ) : (
          /* RESULTS VIEW - CATEGORIZED */
          <div className="animate-in fade-in zoom-in-95 duration-300">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Positive Column */}
              <div className="bg-emerald-50/50 rounded-2xl border border-emerald-100 p-4 min-h-[500px]">
                <div className="flex items-center justify-between mb-4 px-2">
                  <h3 className="font-bold text-emerald-800 flex items-center gap-2">
                    <Smile size={20} className="text-emerald-500" />
                    Positive ({categorizedHistory.Positive.length})
                  </h3>
                </div>
                <div className="space-y-3">
                  {categorizedHistory.Positive.map(item => (
                     <ReviewCard 
                       key={item.id} 
                       item={item} 
                       colorClass="bg-white border-emerald-100 hover:border-emerald-300"
                       badgeClass="bg-emerald-100 text-emerald-700"
                       handleCorrection={handleCorrection}
                       generateSmartReply={generateSmartReply}
                       generatedReplies={generatedReplies}
                       draftingReplyId={draftingReplyId}
                       copyReply={copyReply}
                     />
                  ))}
                  {categorizedHistory.Positive.length === 0 && <EmptyState text="No positive reviews yet." />}
                </div>
              </div>

              {/* Neutral Column */}
              <div className="bg-slate-50/50 rounded-2xl border border-slate-200 p-4 min-h-[500px]">
                <div className="flex items-center justify-between mb-4 px-2">
                  <h3 className="font-bold text-slate-700 flex items-center gap-2">
                    <Meh size={20} className="text-slate-400" />
                    Neutral ({categorizedHistory.Neutral.length})
                  </h3>
                </div>
                <div className="space-y-3">
                  {categorizedHistory.Neutral.map(item => (
                     <ReviewCard 
                       key={item.id} 
                       item={item} 
                       colorClass="bg-white border-slate-200 hover:border-slate-300"
                       badgeClass="bg-slate-100 text-slate-700"
                       handleCorrection={handleCorrection}
                       generateSmartReply={generateSmartReply}
                       generatedReplies={generatedReplies}
                       draftingReplyId={draftingReplyId}
                       copyReply={copyReply}
                     />
                  ))}
                  {categorizedHistory.Neutral.length === 0 && <EmptyState text="No neutral reviews yet." />}
                </div>
              </div>

              {/* Negative Column */}
              <div className="bg-rose-50/50 rounded-2xl border border-rose-100 p-4 min-h-[500px]">
                <div className="flex items-center justify-between mb-4 px-2">
                  <h3 className="font-bold text-rose-800 flex items-center gap-2">
                    <Frown size={20} className="text-rose-500" />
                    Negative ({categorizedHistory.Negative.length})
                  </h3>
                </div>
                <div className="space-y-3">
                  {categorizedHistory.Negative.map(item => (
                     <ReviewCard 
                       key={item.id} 
                       item={item} 
                       colorClass="bg-white border-rose-100 hover:border-rose-300"
                       badgeClass="bg-rose-100 text-rose-700"
                       handleCorrection={handleCorrection}
                       generateSmartReply={generateSmartReply}
                       generatedReplies={generatedReplies}
                       draftingReplyId={draftingReplyId}
                       copyReply={copyReply}
                     />
                  ))}
                  {categorizedHistory.Negative.length === 0 && <EmptyState text="No negative reviews yet." />}
                </div>
              </div>

            </div>
          </div>
        )}
      </main>

      {/* Floating Chatbot */}
      {showChat && (
        <ChatBot history={history} onClose={() => setShowChat(false)} />
      )}
    </div>
  );
}

// Sub-components to keep code clean
const ReviewCard = ({ item, colorClass, badgeClass, handleCorrection, generateSmartReply, generatedReplies, draftingReplyId, copyReply }) => (
  <div className={`p-4 rounded-xl border shadow-sm transition-all group ${colorClass}`}>
    <div className="flex justify-between items-start mb-2">
      <select 
        value={item.label}
        onChange={(e) => handleCorrection(item.id, e.target.value)}
        className={`text-[10px] font-bold px-2 py-0.5 rounded cursor-pointer outline-none border-none uppercase tracking-wide ${badgeClass}`}
      >
        <option value="Positive">Positive</option>
        <option value="Neutral">Neutral</option>
        <option value="Negative">Negative</option>
      </select>
      <span className="text-[10px] text-slate-400">{item.timestamp}</span>
    </div>
    <p className="text-slate-800 text-sm mb-3 leading-relaxed">{item.text}</p>
    
    {/* Smart Reply Actions */}
    <div className="border-t border-dashed border-slate-100 pt-2 flex flex-col gap-2">
      {!generatedReplies[item.id] ? (
        <button 
          onClick={() => generateSmartReply(item)}
          disabled={draftingReplyId === item.id}
          className="self-start text-[10px] flex items-center gap-1.5 text-slate-400 hover:text-indigo-600 font-medium px-2 py-1 hover:bg-slate-50 rounded transition-colors"
        >
            {draftingReplyId === item.id ? (
              <>Generating...</>
            ) : (
              <>
                <Sparkles size={10} />
                Draft Smart Reply
              </>
            )}
        </button>
      ) : (
        <div className="bg-slate-50 rounded p-2 relative group/reply border border-slate-100">
          <p className="text-xs text-slate-600 italic pr-6">{generatedReplies[item.id]}</p>
          <button 
            onClick={() => copyReply(generatedReplies[item.id])}
            className="absolute top-2 right-2 text-slate-400 hover:text-indigo-600"
            title="Copy to clipboard"
          >
            <Copy size={12} />
          </button>
        </div>
      )}
    </div>
  </div>
);

const EmptyState = ({ text }) => (
  <div className="text-center py-10 opacity-40">
    <p className="text-sm text-slate-500 font-medium italic">{text}</p>
  </div>
);