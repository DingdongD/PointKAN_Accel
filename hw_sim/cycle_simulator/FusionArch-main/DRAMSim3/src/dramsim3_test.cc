#include "memory_system.h"
#include <iostream>
#include <list>
#include <string.h>

using namespace std;

// an interface between DRAMSim3 and our ArchTabSim
class DramInterface {
private: 
    map<uint64_t, list<uint64_t> > pendingReadRequests; 
    map<uint64_t, list<uint64_t> > pendingWriteRequests; 
public:
    // Output directory for stats files
    DramInterface(const std::string& config_file, const std::string& output_dir)
        : memory_system_(config_file, output_dir,
              bind(&DramInterface::ReadCallBack, this, std::placeholders::_1),
              bind(&DramInterface::WriteCallBack, this, std::placeholders::_1)),
          clk_(0) {}
    void ReadCallBack(uint64_t addr) { 
        auto p = pendingReadRequests.find(addr);
        assert(p != pendingReadRequests.end() and p->second.size() > 0);

        uint64_t added_cycle = p->second.front();
        uint64_t latency = clk_ - added_cycle;
        pendingReadRequests[addr].pop_front();
        cout << "Read Callback:  0x"<< std::hex << addr << std::dec << " latency="<<latency<<" cycles (end: "<< clk_<< " -> start: "<<added_cycle<<")"<<endl;
    }
    void WriteCallBack(uint64_t addr) { 
        auto p = pendingWriteRequests.find(addr);
        assert(p != pendingWriteRequests.end() and p->second.size() > 0);
        
        uint64_t added_cycle = p->second.front();
        uint64_t latency = clk_ - added_cycle;
        pendingWriteRequests[addr].pop_front();
        cout << "Write Callback: 0x"<< std::hex << addr << std::dec << " latency="<<latency<<" cycles (end: "<< clk_<< " -> start: "<<added_cycle<<")"<<endl;
    }
    void PrintStats() { memory_system_.PrintStats(); }
    void add_pending(uint64_t addr, bool is_write) {
        bool flag = memory_system_.WillAcceptTransaction(addr, is_write);
        if (flag) {
            if (is_write) {
                pendingWriteRequests[addr].push_back(clk_);
            } else {
                pendingReadRequests[addr].push_back(clk_);
            }
            memory_system_.AddTransaction(addr, is_write);
        }
    }
    void cycle() {
        memory_system_.ClockTick();
        clk_++;
    }

protected:
    dramsim3::MemorySystem memory_system_;
    uint64_t clk_;
};

// int main() {
//     DramInterface dram("/home/liuxueyuan/Applications/PointNet/pointnet2/models/arch/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini", "/home/liuxueyuan/Applications/PointNet/pointnet2/models/arch/DRAMSim3/output");
//     printf("Debug Start...\n");
//     dram.add_pending(0, false);
//     dram.add_pending(50, true);
//     dram.add_pending(50, true);
//     dram.add_pending(50, true);
//     dram.add_pending(0, false);
//     dram.add_pending(50, true);
//     for (int i = 0; i < 1000; ++i) {
//         // dram.add_pending(i, true);
//         dram.cycle();
//         dram.PrintStats();
//     }
//     return 0;
// }