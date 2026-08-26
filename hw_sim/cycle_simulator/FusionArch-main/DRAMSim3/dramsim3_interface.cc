#include <iostream>
#include <iomanip>
#include <list>
#include <string.h>
#include <map>
#include <cassert>
#include "Python.h"
#include "memory_system.h"

using namespace std;

enum COMPONENT {
    SAMPLING_ENGINE = 0,
    NEIGHBOR_SEARCH_ENGINE,
    CLUSTERING_ENGINE,
    FEATURE_COMPUTING_ENGINE
};

// request message package format
struct ReqPkg {
    uint64_t added_cycle;
    COMPONENT src;
    ReqPkg(uint64_t added_cycle, COMPONENT src) : added_cycle(added_cycle), src(src) {}
};

// response message package format
struct ResPkg {
    uint64_t addr;
    bool is_write;
    COMPONENT src;
    ResPkg(uint64_t addr, bool is_write, COMPONENT src) : addr(addr), is_write(is_write), src(src) {}
};

// an interface between DRAMSim3 and our ArchTabSim
class DramInterface {
private: 
    map<uint64_t, list<ReqPkg> > pendingReadRequests; 
    map<uint64_t, list<ReqPkg> > pendingWriteRequests;    
    list<ResPkg> retRequests;
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

        uint64_t added_cycle = p->second.front().added_cycle;
        uint64_t latency = clk_ - added_cycle;
        ResPkg retpkg(addr, false, p->second.front().src);
        retRequests.push_back(retpkg);
        pendingReadRequests[addr].pop_front();
        // cout << "Read Callback:  0x"<< std::hex << addr << std::dec << " latency="<<latency<<" cycles (end: "<< clk_<< " -> start: "<<added_cycle<<")"<<endl;
        // cout << "Read Callback: " << std::setw(5) << addr << " latency: " << std::setw(4) << latency << " cycles (end: "<< std::setw(6) << added_cycle << " -> start: "<< std::setw(6) << clk_ << ")" << endl;
    }
    void WriteCallBack(uint64_t addr) { 
        auto p = pendingWriteRequests.find(addr);
        assert(p != pendingWriteRequests.end() and p->second.size() > 0);
        
        uint64_t added_cycle = p->second.front().added_cycle;
        uint64_t latency = clk_ - added_cycle;
        ResPkg retpkg(addr, true, p->second.front().src);
        retRequests.push_back(retpkg);
        pendingWriteRequests[addr].pop_front();
        // cout << "Write Callback: 0x"<< std::hex << addr << std::dec << " latency="<<latency<<" cycles (end: "<< clk_<< " -> start: "<<added_cycle<<")"<<endl;
        // cout << "Write Callback: " << std::setw(5) << addr << " latency: " << std::setw(4) << latency << " cycles (start: "<< std::setw(6) << added_cycle << " -> end: "<< std::setw(6) << clk_ << ")" << endl;
    }
    void PrintStats() { memory_system_.PrintStats(); }
    bool add_pending(uint64_t addr, bool is_write, COMPONENT src) {
        bool flag = memory_system_.WillAcceptTransaction(addr, is_write);
        if (flag) {
            ReqPkg pkg(clk_, src);
            if (is_write) {
                pendingWriteRequests[addr].push_back(pkg);
            } else {
                pendingReadRequests[addr].push_back(pkg);
            }
            memory_system_.AddTransaction(addr, is_write);
        }
        return flag;
    }
    list<ResPkg> cycle() {
        memory_system_.ClockTick();
        clk_++;
        list<ResPkg> tmp = retRequests;     
        retRequests.clear();
        return tmp;
    }

protected:
    dramsim3::MemorySystem memory_system_;
    uint64_t clk_;
};

static DramInterface dram("/home/liuxueyuan/Applications/PointNet/ablation_exp_new/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini", "/home/liuxueyuan/Applications/PointNet/ablation_exp_new/DRAMSim3/output");

static PyObject* add_pending(PyObject* self, PyObject* args) {
    uint64_t addr;
    bool is_write;
    int src;
    if (!PyArg_ParseTuple(args, "Kpi", &addr, &is_write, &src)) {
        return NULL;
    }
    bool flag = dram.add_pending(addr, is_write, (COMPONENT)src);
    return PyBool_FromLong(flag);
}

static PyObject* cycle(PyObject* self, PyObject* args) {
    list<ResPkg> ret = dram.cycle();
    PyObject* ret_list = PyList_New(ret.size());
    int i = 0;
    for (auto& pkg : ret) {
        PyObject* pkg_tuple = PyTuple_New(3);
        PyTuple_SetItem(pkg_tuple, 0, PyLong_FromLong(pkg.addr));
        PyTuple_SetItem(pkg_tuple, 1, PyBool_FromLong(pkg.is_write));
        PyTuple_SetItem(pkg_tuple, 2, PyLong_FromLong(pkg.src));
        PyList_SetItem(ret_list, i, pkg_tuple);
        i++;
    }
    return ret_list;
}

static PyObject* print_stats(PyObject* self, PyObject* args) {
    dram.PrintStats();
    Py_RETURN_NONE;
}

static PyMethodDef dram_methods[] = {
    {"add_pending", add_pending, METH_VARARGS, "add pending request"},
    {"cycle", cycle, METH_VARARGS, "cycle"},
    {"print_stats", print_stats, METH_VARARGS, "print stats"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dram_module = {
    PyModuleDef_HEAD_INIT,
    "dramsim3_interface",
    "dramsim3 interface",
    -1,
    dram_methods
};

PyMODINIT_FUNC PyInit_libdramsim3(void) {
    return PyModule_Create(&dram_module);
}

// int main() {
//     DramInterface dram("/home/liuxueyuan/Applications/PointNet/pointnet2/models/arch/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini", "/home/liuxueyuan/Applications/PointNet/pointnet2/models/arch/DRAMSim3/output");
//     dram.add_pending(3, false, (COMPONENT)1);
    
//     for( int i = 0; i < 1000; i++ ){
//         // dram.add_pending(10, false, (COMPONENT)1);
//         // dram.add_pending(20, true, (COMPONENT)1);
//         list<ResPkg> res = dram.cycle();
//         if (res.size() > 0) {
//             for (auto& pkg : res) {
//                 cout << "addr: " << pkg.addr << " is_write: " << pkg.is_write << " src: " << pkg.src << " cycle: " << i << endl;
//             }
//         }
//     }
// }
