import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [activeTab, setActiveTab] = useState('chat')
  const [apiKey, setApiKey] = useState('')
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('qwen3-coder:latest')
  const [healthStatus, setHealthStatus] = useState(null)
  const [currentApiKey, setCurrentApiKey] = useState('')
  const [stats, setStats] = useState(null)
  const [showPullDialog, setShowPullDialog] = useState(false)
  const [pullModelName, setPullModelName] = useState('')
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [modelMessage, setModelMessage] = useState('')

  // Set up axios interceptor for API key
  useEffect(() => {
    const interceptor = axios.interceptors.request.use((config) => {
      if (apiKey) {
        config.headers.Authorization = `Bearer ${apiKey}`
      }
      return config
    })

    return () => axios.interceptors.request.eject(interceptor)
  }, [apiKey])

  // Simple authentication check
  const handleLogin = (e) => {
    e.preventDefault()
    // TODO: Implement proper backend authentication
    // For now, this is a demo-only UI with no real security
    // DO NOT USE IN PRODUCTION without proper backend auth
    if (username && password) {
      // In production, this should call your backend API for authentication
      setIsAuthenticated(true)
    } else {
      alert('Please enter username and password')
    }
  }

  // Auto-login for testing (remove in production)
  useEffect(() => {
    // Check if we're in development mode or if user wants auto-login
    const urlParams = new URLSearchParams(window.location.search)
    if (urlParams.get('autologin') === 'true') {
      setIsAuthenticated(true)
    }
  }, [])

  // Load initial data
  useEffect(() => {
    if (isAuthenticated) {
      loadHealthStatus()
      loadModels()
      loadCurrentApiKey()
      loadStats()
    }
  }, [isAuthenticated])

  // Auto-refresh stats when Analytics tab is active
  useEffect(() => {
    if (isAuthenticated && activeTab === 'analytics') {
      const interval = setInterval(loadStats, 30000) // 30 seconds
      return () => clearInterval(interval)
    }
  }, [isAuthenticated, activeTab])

  const loadHealthStatus = async () => {
    try {
      const response = await axios.get('/api/health')
      setHealthStatus(response.data)
    } catch (error) {
      setHealthStatus({ status: 'unhealthy', error: error.message })
    }
  }

  const loadModels = async () => {
    try {
      const response = await axios.get('/api/models')
      setModels(response.data.models || [])
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const loadCurrentApiKey = async () => {
    try {
      const response = await axios.get('/api/admin/current-key')
      setCurrentApiKey(response.data.current_api_key)
    } catch (error) {
      console.error('Failed to load API key:', error)
    }
  }

  const generateNewApiKey = async () => {
    try {
      const response = await axios.get('/api/admin/generate-key')
      setCurrentApiKey(response.data.api_key)
      setApiKey(response.data.api_key)
      alert('New API key generated! Make sure to save it.')
    } catch (error) {
      alert('Failed to generate API key: ' + error.message)
    }
  }

  const loadStats = async () => {
    try {
      const response = await axios.get('/api/admin/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const pullModel = async () => {
    if (!pullModelName.trim()) {
      alert('Please enter a model name')
      return
    }

    setIsModelLoading(true)
    setModelMessage('Pulling model... This may take several minutes.')

    try {
      const response = await axios.post(`/api/admin/models/pull?model_name=${encodeURIComponent(pullModelName)}`)
      setModelMessage(`Success: ${response.data.message}`)
      setPullModelName('')
      setShowPullDialog(false)
      // Reload models after successful pull
      setTimeout(() => {
        loadModels()
        setModelMessage('')
      }, 2000)
    } catch (error) {
      setModelMessage(`Error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsModelLoading(false)
    }
  }

  const deleteModel = async (modelName) => {
    if (!confirm(`Are you sure you want to delete ${modelName}? This cannot be undone.`)) {
      return
    }

    setIsModelLoading(true)
    setModelMessage(`Deleting ${modelName}...`)

    try {
      const response = await axios.delete(`/api/admin/models/${encodeURIComponent(modelName)}`)
      setModelMessage(`Success: ${response.data.message}`)
      // Reload models after successful delete
      setTimeout(() => {
        loadModels()
        setModelMessage('')
      }, 2000)
    } catch (error) {
      setModelMessage(`Error: ${error.response?.data?.detail || error.message}`)
      setTimeout(() => setModelMessage(''), 3000)
    } finally {
      setIsModelLoading(false)
    }
  }

  const sendMessage = async () => {
    if (!message.trim()) return

    setIsLoading(true)
    setResponse('')

    try {
      const requestData = {
        message: message,
        model: selectedModel,
        temperature: 0.7,
        max_tokens: 2048
      }

      const response = await axios.post('/api/chat', requestData)
      setResponse(response.data.response)
      setMessage('')
    } catch (error) {
      setResponse(`Error: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="app">
        <div className="login-container">
          <div className="login-form">
            <h1>Local LLM Manager</h1>
            <p>Please sign in to access the management interface</p>
            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  type="text"
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                  className="login-input"
                />
              </div>
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  type="password"
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  className="login-input"
                />
              </div>
              <button type="submit" className="login-button">
                Sign In
              </button>
            </form>
            <div className="login-help">
              <p><strong>Demo UI - No Real Authentication</strong></p>
              <p>⚠️ This is a demonstration UI only</p>
              <p>Enter any username and password to access</p>
              <p><em>For testing: <a href="?autologin=true">Auto-login</a></em></p>
              <p style={{fontSize: '0.8em', color: '#666'}}>
                TODO: Implement proper backend authentication before production use
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Local LLM Manager</h1>
        <div className="header-right">
          <div className="health-status">
            Status: <span className={`status ${healthStatus?.status || 'unknown'}`}>
              {healthStatus?.status || 'Loading...'}
            </span>
            {healthStatus?.models_loaded && (
              <span className="models-count">({healthStatus.models_loaded} models)</span>
            )}
          </div>
          <button 
            onClick={() => setIsAuthenticated(false)}
            className="logout-button"
          >
            Logout
          </button>
        </div>
      </header>

      <nav className="tabs">
        <button 
          className={activeTab === 'chat' ? 'active' : ''} 
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button 
          className={activeTab === 'models' ? 'active' : ''} 
          onClick={() => setActiveTab('models')}
        >
          Models
        </button>
        <button
          className={activeTab === 'settings' ? 'active' : ''}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
        <button
          className={activeTab === 'analytics' ? 'active' : ''}
          onClick={() => setActiveTab('analytics')}
        >
          Analytics
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'chat' && (
          <div className="chat-container">
            <div className="chat-controls">
              <select 
                value={selectedModel} 
                onChange={(e) => setSelectedModel(e.target.value)}
                className="model-select"
              >
                {models.map(model => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({model.size})
                  </option>
                ))}
              </select>
            </div>

            <div className="chat-messages">
              {response && (
                <div className="message response">
                  <strong>Assistant:</strong>
                  <div className="message-content">{response}</div>
                </div>
              )}
            </div>

            <div className="chat-input">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message here... (Press Enter to send)"
                disabled={isLoading}
                rows="3"
              />
              <button 
                onClick={sendMessage} 
                disabled={isLoading || !message.trim()}
                className="send-button"
              >
                {isLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="models-container">
            <div className="models-header">
              <h2>Available Models</h2>
              <button
                onClick={() => setShowPullDialog(true)}
                className="pull-model-button"
                disabled={isModelLoading}
              >
                Pull Model
              </button>
            </div>

            {modelMessage && (
              <div className={`model-message ${modelMessage.includes('Error') ? 'error' : 'success'}`}>
                {modelMessage}
              </div>
            )}

            <div className="models-list">
              {models.map(model => (
                <div key={model.name} className="model-card">
                  <h3>{model.name}</h3>
                  <p>Size: {model.size}</p>
                  <p>Status: <span className="status available">{model.status}</span></p>
                  <p>Modified: {model.modified}</p>
                  <button
                    onClick={() => deleteModel(model.name)}
                    className="delete-model-button"
                    disabled={isModelLoading}
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>

            {showPullDialog && (
              <div className="modal-overlay" onClick={() => !isModelLoading && setShowPullDialog(false)}>
                <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                  <h3>Pull Model from Ollama Library</h3>
                  <input
                    type="text"
                    value={pullModelName}
                    onChange={(e) => setPullModelName(e.target.value)}
                    placeholder="Enter model name (e.g., llama2:latest, tinyllama:latest)"
                    disabled={isModelLoading}
                    onKeyPress={(e) => e.key === 'Enter' && pullModel()}
                  />
                  <div className="modal-buttons">
                    <button
                      onClick={pullModel}
                      disabled={isModelLoading || !pullModelName.trim()}
                      className="pull-button"
                    >
                      {isModelLoading ? 'Pulling...' : 'Pull'}
                    </button>
                    <button
                      onClick={() => setShowPullDialog(false)}
                      disabled={isModelLoading}
                      className="cancel-button"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="settings-container">
            <h2>API Settings</h2>
            
            <div className="api-key-section">
              <h3>Current API Key</h3>
              <div className="api-key-display">
                <code>{currentApiKey || 'No API key generated'}</code>
              </div>
              <button onClick={generateNewApiKey} className="generate-key-btn">
                Generate New API Key
              </button>
            </div>

            <div className="api-key-input">
              <h3>Set API Key for Current Session</h3>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your API key here"
                className="api-key-input-field"
              />
              <p className="help-text">
                This key will be used for all API requests in this session
              </p>
            </div>

            <div className="usage-instructions">
              <h3>Usage Instructions</h3>
              <p>Use the API key in the Authorization header:</p>
              <code>Authorization: Bearer YOUR_API_KEY</code>
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="analytics-container">
            <h2>System Analytics</h2>

            {stats ? (
              <>
                <div className="stats-cards">
                  <div className="stat-card">
                    <h3>Total Requests</h3>
                    <div className="stat-value">{stats.total_requests}</div>
                    <div className="stat-label">All time</div>
                  </div>
                  <div className="stat-card">
                    <h3>Avg Response Time</h3>
                    <div className="stat-value">{stats.avg_response_time}s</div>
                    <div className="stat-label">Per request</div>
                  </div>
                  <div className="stat-card">
                    <h3>Cache Hit Ratio</h3>
                    <div className="stat-value">{(stats.cache_hit_ratio * 100).toFixed(1)}%</div>
                    <div className="stat-label">{stats.redis_hits} hits / {stats.redis_hits + stats.redis_misses} total</div>
                  </div>
                  <div className="stat-card">
                    <h3>Models Loaded</h3>
                    <div className="stat-value">{stats.models_loaded}</div>
                    <div className="stat-label">Available models</div>
                  </div>
                </div>

                <div className="charts-grid">
                  <div className="chart-container">
                    <h3>Requests by Endpoint</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={Object.entries(stats.requests_by_endpoint).map(([name, count]) => ({ name, count }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {stats.top_models && stats.top_models.length > 0 && (
                    <div className="chart-container">
                      <h3>Top Models by Usage</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={stats.top_models}
                            dataKey="count"
                            nameKey="model"
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            label
                          >
                            {stats.top_models.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'][index % 5]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                <div className="stats-footer">
                  <p>Last updated: {new Date(stats.timestamp).toLocaleString()}</p>
                  <button onClick={loadStats} className="refresh-button">Refresh Now</button>
                </div>
              </>
            ) : (
              <div className="loading">Loading analytics...</div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
