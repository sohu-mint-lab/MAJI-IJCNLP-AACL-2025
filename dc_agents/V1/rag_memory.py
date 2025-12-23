# rag_memory.py
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from .modelsv1 import ConversationTurn 
import os

class RAGMemorySystem:
    """RAG-based conversation memory system"""
    
    def __init__(self, db_path: str = "data/rag/interview_memory.db", model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.init_database()
        
    def init_database(self):
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                round_num INTEGER,
                speaker TEXT,
                content TEXT,
                timestamp TEXT,
                embedding BLOB
            )
        ''')
        
        # Topic coverage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                topic_id TEXT,
                topic_name TEXT,
                depth_level INTEGER,
                coverage_score REAL,
                last_updated TEXT
            )
        ''')
        
        # Emotional state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                round_num INTEGER,
                emotion TEXT,
                intensity REAL,
                context TEXT,
                timestamp TEXT
            )
        ''')
        
        # Follow-up opportunities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS followup_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                round_num INTEGER,
                opportunity_type TEXT,
                description TEXT,
                priority REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT
            )
        ''')
        # Long-term memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                round_num INTEGER,
                snippet TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_conversation_turn(self, session_id: str, round_num: int, turn: ConversationTurn):
        """Add a conversation turn"""
        try:
            # Generate embedding
            embedding = self.model.encode([turn.text])[0]
            embedding_blob = embedding.tobytes()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (session_id, round_num, speaker, content, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, round_num, turn.speaker, turn.text, datetime.now().isoformat(), embedding_blob))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error adding conversation turn: {e}")
        
    def update_topic_coverage(self, session_id: str, topic_id: str, topic_name: str, 
                            depth_level: int, coverage_score: float):
        """Update topic coverage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if it already exists
            cursor.execute('''
                SELECT id FROM topic_coverage 
                WHERE session_id = ? AND topic_id = ?
            ''', (session_id, topic_id))
            
            if cursor.fetchone():
                # Update existing record
                cursor.execute('''
                    UPDATE topic_coverage 
                    SET depth_level = ?, coverage_score = ?, last_updated = ?
                    WHERE session_id = ? AND topic_id = ?
                ''', (depth_level, coverage_score, datetime.now().isoformat(), session_id, topic_id))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO topic_coverage (session_id, topic_id, topic_name, depth_level, coverage_score, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, topic_id, topic_name, depth_level, coverage_score, datetime.now().isoformat()))
                
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating topic coverage: {e}")
        
    def add_emotional_state(self, session_id: str, round_num: int, emotion: str, 
                          intensity: float = 1.0, context: str = ""):
        """Add emotional state"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emotional_states (session_id, round_num, emotion, intensity, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, round_num, emotion, intensity, context, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error adding emotional state: {e}")
        
    def add_followup_opportunity(self, session_id: str, round_num: int, 
                               opportunity_type: str, description: str, priority: float = 0.5):
        """Add follow-up opportunity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO followup_opportunities (session_id, round_num, opportunity_type, description, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, round_num, opportunity_type, description, priority, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error adding follow-up opportunity: {e}")
        
    def search_similar_conversations(self, query: str, session_id: str, 
                                   top_k: int = 5) -> List[Dict]:
        """Search similar conversations"""
        try:
            query_embedding = self.model.encode([query])[0]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, round_num, speaker, content, embedding
                FROM conversations 
                WHERE session_id = ?
                ORDER BY round_num DESC
            ''', (session_id,))
            
            results = []
            for row in cursor.fetchall():
                conv_id, round_num, speaker, content, embedding_blob = row
                if embedding_blob:  # Ensure embedding exists
                    conv_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    similarity = cosine_similarity([query_embedding], [conv_embedding])[0][0]
                    
                    results.append({
                        'id': conv_id,
                        'round_num': round_num,
                        'speaker': speaker,
                        'content': content,
                        'similarity': similarity
                    })
                
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            conn.close()
            
            return results[:top_k]
        except Exception as e:
            print(f"Error searching similar conversations: {e}")
            return []
        
    def get_recent_conversations(self, session_id: str, limit: int = 6) -> List[Dict]:
        """Get recent conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT round_num, speaker, content
                FROM conversations 
                WHERE session_id = ?
                ORDER BY round_num DESC
                LIMIT ?
            ''', (session_id, limit))
            
            results = []
            for row in cursor.fetchall():
                round_num, speaker, content = row
                results.append({
                    'round_num': round_num,
                    'speaker': speaker,
                    'content': content
                })
                
            conn.close()
            return list(reversed(results))  # Return in chronological order
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
        
    def get_topic_coverage_summary(self, session_id: str) -> Dict:
        """Get topic coverage summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT topic_id, topic_name, depth_level, coverage_score
                FROM topic_coverage 
                WHERE session_id = ?
                ORDER BY coverage_score DESC
            ''', (session_id,))
            
            topics = {}
            for row in cursor.fetchall():
                topic_id, topic_name, depth_level, coverage_score = row
                topics[topic_id] = {
                    'name': topic_name,
                    'depth': depth_level,
                    'coverage': coverage_score
                }
                
            conn.close()
            return topics
        except Exception as e:
            print(f"Error getting topic coverage summary: {e}")
            return {}
        
    def get_emotional_trajectory(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get emotional trajectory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT round_num, emotion, intensity, context
                FROM emotional_states 
                WHERE session_id = ?
                ORDER BY round_num DESC
                LIMIT ?
            ''', (session_id, limit))
            
            emotions = []
            for row in cursor.fetchall():
                round_num, emotion, intensity, context = row
                emotions.append({
                    'round': round_num,
                    'emotion': emotion,
                    'intensity': intensity,
                    'context': context
                })
                
            conn.close()
            return list(reversed(emotions))
        except Exception as e:
            print(f"Error getting emotional trajectory: {e}")
            return []
        
    def get_pending_followups(self, session_id: str, top_k: int = 3) -> List[Dict]:
        """Get pending follow-up opportunities"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT round_num, opportunity_type, description, priority
                FROM followup_opportunities 
                WHERE session_id = ? AND status = 'pending'
                ORDER BY priority DESC, round_num DESC
                LIMIT ?
            ''', (session_id, top_k))
            
            followups = []
            for row in cursor.fetchall():
                round_num, opp_type, description, priority = row
                followups.append({
                    'round': round_num,
                    'type': opp_type,
                    'description': description,
                    'priority': priority
                })
                
            conn.close()
            return followups
        except Exception as e:
            print(f"Error getting pending follow-ups: {e}")
            return []
        
    def mark_followup_used(self, session_id: str, round_num: int, opportunity_type: str):
        """Mark follow-up opportunity as used"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE followup_opportunities 
                SET status = 'used'
                WHERE session_id = ? AND round_num = ? AND opportunity_type = ?
            ''', (session_id, round_num, opportunity_type))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error marking follow-up opportunity: {e}")
        
    def get_conversation_statistics(self, session_id: str) -> Dict:
        """Get conversation statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total turns
            cursor.execute('SELECT COUNT(*) FROM conversations WHERE session_id = ?', (session_id,))
            total_turns = cursor.fetchone()[0]
            
            # Topic count
            cursor.execute('SELECT COUNT(*) FROM topic_coverage WHERE session_id = ?', (session_id,))
            topic_count = cursor.fetchone()[0]
            
            # Emotional state count
            cursor.execute('SELECT COUNT(*) FROM emotional_states WHERE session_id = ?', (session_id,))
            emotion_count = cursor.fetchone()[0]
            
            # Pending follow-ups count
            cursor.execute('''
                SELECT COUNT(*) FROM followup_opportunities 
                WHERE session_id = ? AND status = 'pending'
            ''', (session_id,))
            pending_followups = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_turns': total_turns,
                'topics_covered': topic_count,
                'emotional_events': emotion_count,
                'pending_followups': pending_followups
            }
        except Exception as e:
            print(f"Error getting conversation statistics: {e}")
            return {
                'total_turns': 0,
                'topics_covered': 0,
                'emotional_events': 0,
                'pending_followups': 0
            }
        
    def cleanup_old_sessions(self, days_old: int = 30):
        """Cleanup old session data"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old records
            cursor.execute('DELETE FROM conversations WHERE timestamp < ?', (cutoff_iso,))
            cursor.execute('DELETE FROM topic_coverage WHERE last_updated < ?', (cutoff_iso,))
            cursor.execute('DELETE FROM emotional_states WHERE timestamp < ?', (cutoff_iso,))
            cursor.execute('DELETE FROM followup_opportunities WHERE created_at < ?', (cutoff_iso,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error cleaning up old session data: {e}")
        
    def export_session_data(self, session_id: str, output_file: str):
        """Export session data"""
        try:
            # Ensure output_file is in data/rag if no directory is specified
            if os.path.dirname(output_file) == '':
                output_file = os.path.join('data/rag', output_file)
            conn = sqlite3.connect(self.db_path)
            
            # Get all related data
            cursor = conn.cursor()
            
            # Conversation history
            cursor.execute('''
                SELECT id, session_id, round_num, speaker, content, timestamp
                FROM conversations WHERE session_id = ? ORDER BY round_num
            ''', (session_id,))
            conversations = []
            conv_columns = ['id', 'session_id', 'round_num', 'speaker', 'content', 'timestamp']
            for row in cursor.fetchall():
                conversations.append(dict(zip(conv_columns, row)))
            
            # Topic coverage
            cursor.execute('''
                SELECT id, session_id, topic_id, topic_name, depth_level, coverage_score, last_updated
                FROM topic_coverage WHERE session_id = ?
            ''', (session_id,))
            topics = []
            topic_columns = ['id', 'session_id', 'topic_id', 'topic_name', 'depth_level', 'coverage_score', 'last_updated']
            for row in cursor.fetchall():
                topics.append(dict(zip(topic_columns, row)))
            
            # Emotional state
            cursor.execute('''
                SELECT id, session_id, round_num, emotion, intensity, context, timestamp
                FROM emotional_states WHERE session_id = ? ORDER BY round_num
            ''', (session_id,))
            emotions = []
            emotion_columns = ['id', 'session_id', 'round_num', 'emotion', 'intensity', 'context', 'timestamp']
            for row in cursor.fetchall():
                emotions.append(dict(zip(emotion_columns, row)))
            
            # Follow-up opportunities
            cursor.execute('''
                SELECT id, session_id, round_num, opportunity_type, description, priority, status, created_at
                FROM followup_opportunities WHERE session_id = ? ORDER BY round_num
            ''', (session_id,))
            followups = []
            followup_columns = ['id', 'session_id', 'round_num', 'opportunity_type', 'description', 'priority', 'status', 'created_at']
            for row in cursor.fetchall():
                followups.append(dict(zip(followup_columns, row)))
            
            conn.close()
            
            # Organize data
            export_data = {
                'session_id': session_id,
                'export_timestamp': datetime.now().isoformat(),
                'conversations': conversations,
                'topic_coverage': topics,
                'emotional_states': emotions,
                'followup_opportunities': followups
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error exporting session data: {e}")
   
    def add_memory_snippet(self, session_id: str, round_num: int, snippet: str):
        """Persist memory snippets"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO long_term_memory (session_id, round_num, snippet, created_at)
                VALUES (?, ?, ?, ?)
            ''', (session_id, round_num, snippet, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error adding long-term memory snippet: {e}")

    def get_long_term_memory(self, session_id: str, limit: int = 10) -> List[str]:
        """Read the last N long-term memories (reverse chronological order)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT snippet FROM long_term_memory
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ''', (session_id, limit))
            rows = cursor.fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"Error reading long-term memory: {e}")
            return []
