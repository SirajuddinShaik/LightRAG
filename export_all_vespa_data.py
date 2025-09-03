#!/usr/bin/env python3
"""
Export ALL Vespa data to JSON format

This script exports all documents from Vespa without any document limits,
providing complete data export functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from vespa_integration import (
    export_all_vespa_data_to_json,
    export_vespa_data_by_type,
    convert_vespa_to_json_lines,
    get_vespa_connection,
    VespaJSONExporter
)

async def export_complete_vespa_dataset():
    """Export ALL Vespa data without document limits"""
    print("🚀 Starting COMPLETE Vespa Data Export")
    print("=" * 60)
    print("⚠️  This will export ALL documents from Vespa")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Export ALL documents to a single comprehensive JSON file
        print("\n1️⃣ Exporting ALL documents to comprehensive JSON...")
        all_data_stats = await export_all_vespa_data_to_json(
            output_file=f"vespa_complete_export_{timestamp}.json",
            schema="mail",
            max_documents=None,  # No limit - export everything
            include_metadata=True
        )
        
        if all_data_stats.get('success'):
            print(f"✅ Complete export successful:")
            print(f"   📄 Total documents: {all_data_stats['total_documents']}")
            print(f"   📁 File: {all_data_stats['output_file']}")
            print(f"   💾 File size: {all_data_stats['file_size_bytes']:,} bytes")
            print(f"   📅 Export time: {all_data_stats['export_timestamp']}")
        else:
            print(f"❌ Complete export failed: {all_data_stats.get('error')}")
            return

        # 2. Export ALL documents to JSON Lines format (for streaming/processing)
        print("\n2️⃣ Exporting ALL documents to JSON Lines format...")
        jsonl_stats = await convert_vespa_to_json_lines(
            output_file=f"vespa_complete_export_{timestamp}.jsonl",
            schema="mail",
            max_documents=None  # No limit
        )
        
        if jsonl_stats.get('success'):
            print(f"✅ JSON Lines export successful:")
            print(f"   📄 Total documents: {jsonl_stats['total_documents']}")
            print(f"   📁 File: {jsonl_stats['output_file']}")
            print(f"   💾 File size: {jsonl_stats['file_size_bytes']:,} bytes")
        else:
            print(f"❌ JSON Lines export failed: {jsonl_stats.get('error')}")

        # 3. Export by document types (separate files for each type)
        print("\n3️⃣ Exporting documents by type (separate files)...")
        type_export_stats = await export_vespa_data_by_type(
            doc_types=['email', 'document', 'chat', 'meeting_notes'],
            output_dir=f"vespa_by_type_{timestamp}",
            schema="mail",
            max_documents_per_type=None  # No limit per type
        )
        
        if type_export_stats.get('success'):
            print(f"✅ Type-based export successful:")
            print(f"   📁 Directory: {type_export_stats['output_directory']}")
            print(f"   📊 Types processed: {type_export_stats['total_types_processed']}")
            
            for doc_type, result in type_export_stats['results_by_type'].items():
                if result['success']:
                    print(f"   📄 {doc_type}: {result['document_count']:,} documents, "
                          f"{result['file_size_bytes']:,} bytes")
                else:
                    print(f"   ⚠️ {doc_type}: {result['error']}")
        else:
            print(f"❌ Type-based export failed: {type_export_stats.get('error')}")

        # 4. Create a lightweight summary file
        print("\n4️⃣ Creating lightweight summary...")
        summary_stats = await export_vespa_lightweight_json(
            output_file=f"vespa_summary_{timestamp}.json",
            schema="mail",
            max_documents=None,
            fields=['id', 'title', 'doc_type', 'timestamp', 'source']
        )
        
        if summary_stats.get('success'):
            print(f"✅ Summary export successful:")
            print(f"   📄 Total documents: {summary_stats['total_documents']}")
            print(f"   📁 File: {summary_stats['output_file']}")
            print(f"   💾 File size: {summary_stats['file_size_bytes']:,} bytes")
            print(f"   📝 Fields: {summary_stats['included_fields']}")

        # 5. Generate export report
        print("\n" + "=" * 60)
        print("📊 EXPORT SUMMARY REPORT")
        print("=" * 60)
        
        if all_data_stats.get('success'):
            total_docs = all_data_stats['total_documents']
            total_size = all_data_stats['file_size_bytes']
            
            print(f"🎯 Total Documents Exported: {total_docs:,}")
            print(f"💾 Total Data Size: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
            print(f"📅 Export Timestamp: {timestamp}")
            print(f"\n📁 Generated Files:")
            print(f"   - vespa_complete_export_{timestamp}.json (Complete dataset)")
            print(f"   - vespa_complete_export_{timestamp}.jsonl (JSON Lines format)")
            print(f"   - vespa_summary_{timestamp}.json (Lightweight summary)")
            print(f"   - vespa_by_type_{timestamp}/ (Directory with type-separated files)")
            
            # Calculate statistics
            avg_doc_size = total_size / total_docs if total_docs > 0 else 0
            print(f"\n📈 Statistics:")
            print(f"   - Average document size: {avg_doc_size:.0f} bytes")
            print(f"   - Export completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"\n🎉 SUCCESS: All Vespa data exported successfully!")
            
        else:
            print("❌ Export failed - check error messages above")

    except Exception as e:
        print(f"💥 Critical error during export: {e}")
        import traceback
        traceback.print_exc()

async def export_vespa_lightweight_json(output_file, schema, max_documents, fields):
    """Helper function for lightweight export"""
    from vespa_integration import export_vespa_lightweight_json as export_light
    return await export_light(
        output_file=output_file,
        schema=schema,
        max_documents=max_documents,
        fields=fields
    )

async def verify_export_integrity(json_file: str):
    """Verify the integrity of exported JSON data"""
    print(f"\n🔍 Verifying export integrity for {json_file}...")
    
    try:
        if not Path(json_file).exists():
            print(f"❌ File {json_file} does not exist")
            return False
            
        # Load and verify JSON structure
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required structure
        if 'export_metadata' not in data:
            print("❌ Missing export_metadata section")
            return False
            
        if 'documents' not in data:
            print("❌ Missing documents section")
            return False
            
        doc_count = len(data['documents'])
        metadata = data['export_metadata']
        
        print(f"✅ JSON structure valid")
        print(f"   📄 Documents in file: {doc_count:,}")
        print(f"   📊 Metadata count: {metadata.get('total_documents', 'N/A')}")
        print(f"   📅 Export timestamp: {metadata.get('export_timestamp', 'N/A')}")
        print(f"   🏷️ Schema: {metadata.get('schema', 'N/A')}")
        
        # Verify a sample document structure
        if doc_count > 0:
            sample_doc = data['documents'][0]
            required_fields = ['id', 'title', 'content', 'doc_type']
            
            missing_fields = [field for field in required_fields if field not in sample_doc]
            if missing_fields:
                print(f"⚠️ Sample document missing fields: {missing_fields}")
            else:
                print(f"✅ Sample document structure valid")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    print("🌟 Vespa Complete Data Export Tool")
    print("=" * 60)
    print("This tool exports ALL documents from Vespa to JSON formats")
    print("⚠️  Warning: This may take time for large datasets")
    print("=" * 60)
    
    # Run the complete export
    asyncio.run(export_complete_vespa_dataset())
    
    # Verify the latest export
    import glob
    json_files = glob.glob("vespa_complete_export_*.json")
    if json_files:
        latest_file = max(json_files)  # Get the most recent file
        asyncio.run(verify_export_integrity(latest_file))
    
    print("\n📋 Export complete! All Vespa data has been exported to JSON formats.")
