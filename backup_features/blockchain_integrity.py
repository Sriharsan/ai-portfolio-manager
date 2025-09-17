import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

@dataclass
class PortfolioDecision:
    """Represents a portfolio optimization decision with full audit trail."""
    timestamp: str
    portfolio_id: str
    optimization_method: str
    assets: Dict[str, float]  # {symbol: weight}
    risk_metrics: Dict[str, float]
    model_version: str
    data_sources: List[str]
    decision_hash: str = None

class MerkleTreeBuilder:
    """
    Builds Merkle trees for batch verification of portfolio decisions.
    Enables efficient verification of large numbers of decisions.
    """
    
    def __init__(self):
        self.leaves = []
    
    def add_decision(self, decision: PortfolioDecision) -> str:
        """Add a portfolio decision and return its hash."""
        decision_json = json.dumps(asdict(decision), sort_keys=True)
        decision_hash = hashlib.sha256(decision_json.encode()).hexdigest()
        decision.decision_hash = decision_hash
        self.leaves.append(decision_hash)
        return decision_hash
    
    def build_merkle_root(self) -> str:
        """Build Merkle tree and return root hash."""
        if not self.leaves:
            return ""
        
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            # Ensure even number of nodes
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            
            # Build parent level
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)
            
            current_level = next_level
        
        return current_level[0]

class BlockchainAnchor:
    """
    Simulated blockchain anchoring service.
    In production, would integrate with Hyperledger Fabric or Ethereum.
    """
    
    def __init__(self):
        self.chain = []
        self.private_key = self._generate_key_pair()[0]
        
    def _generate_key_pair(self) -> Tuple:
        """Generate RSA key pair for signing."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def anchor_merkle_root(self, merkle_root: str, batch_id: str) -> Dict:
        """
        Anchor Merkle root to blockchain.
        Returns transaction details for verification.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Create block data
        block_data = {
            'batch_id': batch_id,
            'merkle_root': merkle_root,
            'timestamp': timestamp,
            'block_number': len(self.chain) + 1
        }
        
        # Sign the block
        block_json = json.dumps(block_data, sort_keys=True)
        signature = self.private_key.sign(
            block_json.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Add to chain
        block = {
            **block_data,
            'signature': signature.hex()
        }
        self.chain.append(block)
        
        return {
            'transaction_id': f"tx_{len(self.chain):08d}",
            'block_number': block['block_number'],
            'merkle_root': merkle_root,
            'timestamp': timestamp
        }
    
    def verify_integrity(self, merkle_root: str, transaction_id: str) -> bool:
        """Verify that a Merkle root exists in the blockchain."""
        for block in self.chain:
            if block['merkle_root'] == merkle_root:
                return True
        return False

# Example usage for interview demonstration
def demonstrate_blockchain_audit():
    """
    Demonstrate complete audit trail workflow.
    Shows how portfolio decisions are recorded immutably.
    """
    # Initialize components
    merkle_builder = MerkleTreeBuilder()
    blockchain = BlockchainAnchor()
    
    # Create sample portfolio decisions
    decisions = [
        PortfolioDecision(
            timestamp=datetime.utcnow().isoformat(),
            portfolio_id="INST_001",
            optimization_method="Black-Litterman",
            assets={"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15, "TSLA": 0.10, "SPY": 0.30},
            risk_metrics={"sharpe": 1.24, "var_95": 0.048, "max_drawdown": 0.125},
            model_version="v2.1.0",
            data_sources=["AlphaVantage", "FRED", "NewsAPI"]
        ),
        PortfolioDecision(
            timestamp=datetime.utcnow().isoformat(),
            portfolio_id="INST_002",
            optimization_method="Risk-Parity",
            assets={"BND": 0.40, "VTI": 0.30, "VEA": 0.20, "VWO": 0.10},
            risk_metrics={"sharpe": 0.89, "var_95": 0.032, "max_drawdown": 0.089},
            model_version="v2.1.0",
            data_sources=["AlphaVantage", "FRED"]
        )
    ]
    
    # Add decisions to Merkle tree
    for decision in decisions:
        merkle_builder.add_decision(decision)
    
    # Build Merkle root and anchor to blockchain
    merkle_root = merkle_builder.build_merkle_root()
    batch_id = f"BATCH_{int(time.time())}"
    
    anchor_result = blockchain.anchor_merkle_root(merkle_root, batch_id)
    
    print("Blockchain Audit Trail Demonstration:")
    print(f"Merkle Root: {merkle_root}")
    print(f"Transaction ID: {anchor_result['transaction_id']}")
    print(f"Block Number: {anchor_result['block_number']}")
    
    # Verify integrity
    is_valid = blockchain.verify_integrity(merkle_root, anchor_result['transaction_id'])
    print(f"Integrity Verified: {is_valid}")
