import { useState } from "react";

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");
  const [citations, setCitations] = useState(null);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) {
      alert("Please enter a question");
      return;
    }

    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

    setLoading(true);
    setAnswer("");
    setContext("");
    setCitations(null);

    try {
      const res =await fetch(`${BACKEND_URL}/qa`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});

      const data = await res.json();

      setAnswer(data.answer || "No answer returned");
      setContext(data.context || "");
      setCitations(data.citations || null);
    } catch (err) {
      alert("Backend not reachable");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-5">
      {/* Header */}
      <h1 className="mb-4 d-flex align-items-center gap-2">
        <i className="bi bi-diagram-3-fill text-primary fs-3"></i>
        IKMS Multi-Agent RAG System
      </h1>

      {/* Question Input */}
      <div className="mb-3">
        <label className="form-label d-flex align-items-center gap-2">
          <i className="bi bi-question-circle-fill"></i>
          Ask a question
        </label>

        <textarea
          className="form-control"
          rows="4"
          placeholder="Ask a question about the document..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
      </div>

      {/* Ask Button */}
      <button
        className="btn btn-primary d-flex align-items-center gap-2"
        onClick={askQuestion}
        disabled={loading}
      >
        {loading ? (
          <>
            <span className="spinner-border spinner-border-sm"></span>
            Processing...
          </>
        ) : (
          <>
            <i className="bi bi-send-fill"></i>
            Ask
          </>
        )}
      </button>

      {/* Response */}
      {answer && (
        <div className="card mt-4 shadow-sm">
          <div className="card-body">
            <h5 className="card-title d-flex align-items-center gap-2 text-success">
              <i className="bi bi-chat-left-text-fill"></i>
              Response
            </h5>
            <p className="card-text">{answer}</p>
          </div>
        </div>
      )}

      {/* Context */}
      {context && (
        <div className="card mt-3 shadow-sm">
          <div className="card-body">
            <h5 className="card-title d-flex align-items-center gap-2 text-secondary">
              <i className="bi bi-file-earmark-text-fill"></i>
              Context
            </h5>
            <pre className="bg-light p-3 rounded small mb-0">
              {context}
            </pre>
          </div>
        </div>
      )}

      {/* Citations */}
      {citations && (
        <div className="card mt-3 shadow-sm">
          <div className="card-body">
            <h5 className="card-title d-flex align-items-center gap-2">
              <i className="bi bi-link-45deg"></i>
              Citations
            </h5>

            {Object.entries(citations).map(([key, value]) => (
              <div
                key={key}
                className="border-start border-3 border-primary ps-3 mb-3"
              >
                <strong>{key}</strong>
                <div>Page: {value.page}</div>
                <div>Source: {value.source}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
