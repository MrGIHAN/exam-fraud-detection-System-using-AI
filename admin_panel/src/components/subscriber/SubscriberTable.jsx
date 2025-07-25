import React, { useEffect, useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { BsArrowLeft, BsArrowRight, BsDot } from 'react-icons/bs';
import { Link } from 'react-router-dom';
import { Modal, Button, Table } from 'react-bootstrap';

const SubscriberTable = ({ title }) => {
    const [sessions, setSessions] = useState([]);
    const [searchId, setSearchId] = useState('');
    const [summaryModalOpen, setSummaryModalOpen] = useState(false);
    const [summaryData, setSummaryData] = useState(null);
    const [selectedSession, setSelectedSession] = useState(null);

    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedSessions = sessions.slice(startIndex, endIndex);


    const [cheatFlags, setCheatFlags] = useState({});   
const [flagsLoading, setFlagsLoading] = useState(false);

const getSessionFlag = async (session) => {
  try {
    const res = await axios.post(`${BASE_URL}/examSession/sessionSummaryByModelType`, {
      sessionId: session._id,
      studentId: session.studentId, 
    });
    return !!res.data?.cheatingThresholdExceeded;
  } catch {
    return false; 
  }
};

    useEffect(() => {
        fetchSessions();
    }, [refreshKey]);

    const fetchSessions = async () => {
        try {
            const res = await axios.get(`${BASE_URL}/student/getSessionsByCustomStudentId`, {
                params: searchId ? { studentId: searchId } : {},
            });
            setSessions(res.data);
            setCheatFlags({}); 
    setCurrentPage(1); 
        } catch (err) {
            console.error('Failed to load sessions', err);
        }
    };

    const handleSearch = (e) => setSearchId(e.target.value);
    const handleSearchSubmit = (e) => {
        e.preventDefault();
        fetchSessions();
    };



    useEffect(() => {
  const loadPageFlags = async () => {
    if (paginatedSessions.length === 0) return;

    
    const toFetch = paginatedSessions.filter(s => !(s._id in cheatFlags));
    if (toFetch.length === 0) return;

    setFlagsLoading(true);
    try {
      
      const results = await Promise.allSettled(toFetch.map(getSessionFlag));
      const updates = {};
      results.forEach((r, i) => {
        const sid = toFetch[i]._id;
        updates[sid] = r.status === 'fulfilled' ? r.value : false;
      });
      setCheatFlags(prev => ({ ...prev, ...updates }));
    } finally {
      setFlagsLoading(false);
    }
  };

  loadPageFlags();
}, [sessions, currentPage]); 




    const openSummaryModal = async (session) => {
        try {
            setSelectedSession(session);
            const res = await axios.post(`${BASE_URL}/examSession/sessionSummaryByModelType`, {
                sessionId: session._id,
                studentId: session.studentId
            });
            setSummaryData(res.data);
            setSummaryModalOpen(true);
        } catch (error) {
            console.error('Failed to load summary', error);
        }
    };

    const closeModal = () => {
        setSummaryModalOpen(false);
        setSummaryData(null);
        setSelectedSession(null);
    };

    if (isRemoved) return null;

    return (
        <div className="col-xxl-12">
            <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
                <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
                <div className="card-body p-3">
                    <form onSubmit={handleSearchSubmit} className="mb-3 d-flex gap-2">
                        <input
                            type="text"
                            className="form-control"
                            placeholder="Search by Student ID (e.g. STUE1A4)"
                            value={searchId}
                            onChange={handleSearch}
                        />
                        <button type="submit" className="btn btn-primary">Search</button>
                        <button type="button" className="btn btn-secondary" onClick={() => { setSearchId(''); fetchSessions(); }}>
                            Clear
                        </button>
                    </form>
                    <div className="table-responsive">
                        <table className="table table-hover">
                            <thead>
                                <tr>
                                    <th>Student ID</th>
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Started</th>
                                    <th>Ended</th>
                                    <th>Duration (sec)</th>
                                    <th>Summary</th>
                                </tr>
                            </thead>
                            <tbody>
                                    {paginatedSessions.map((s) => {
                                      const exceeded = cheatFlags[s._id]; 
                                    
                                      return (
                                        <tr
                                          key={s._id}
                                          className={exceeded ? 'text-danger fw-semibold' : ''}

                                        >
                                          <td className={exceeded ? 'text-danger fw-semibold' : ''}>
                                            {s.studentCustomId}
                                            {flagsLoading && !(s._id in cheatFlags) ? (
                                              <span className="ms-2 spinner-border spinner-border-sm" />
                                            ) : null}
                                          </td>
                                          <td>{s.name}</td>
                                          <td>{s.email}</td>
                                          <td>{new Date(s.startedAt).toLocaleString()}</td>
                                          <td>{s.endedAt ? new Date(s.endedAt).toLocaleString() : '-'}</td>
                                          <td>{s.duration ?? '-'}</td>
                                          <td>
                                            <button
                                              className={`btn btn-sm ${exceeded ? 'btn-danger' : 'btn-outline-primary'}`}
                                              onClick={() => openSummaryModal(s)}
                                            >
                                              View
                                            </button>
                                          </td>
                                        </tr>
                                      );
                                    })}
                            </tbody>

                        </table>
                    </div>
                </div>

                {/* Pagination */}
                <div className="card-footer">
                    <ul className="list-unstyled d-flex align-items-center gap-2 mb-0 pagination-common-style">
                        <li>
                            <Link to="#" onClick={() => setCurrentPage(p => Math.max(p - 1, 1))} className={currentPage === 1 ? 'disabled' : ''}>
                                <BsArrowLeft size={16} />
                            </Link>
                        </li>
                        {Array.from({ length: Math.ceil(sessions.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(sessions.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(sessions.length / itemsPerPage) - 1)) {
                                return (
                                    <li key={`dots-${index}`}>
                                        <Link to="#" onClick={(e) => e.preventDefault()}>
                                            <BsDot size={16} />
                                        </Link>
                                    </li>
                                );
                            }
                            return shouldShow ? (
                                <li key={index}>
                                    <Link
                                        to="#"
                                        onClick={() => setCurrentPage(page)}
                                        className={currentPage === page ? 'active' : ''}
                                    >
                                        {page}
                                    </Link>
                                </li>
                            ) : null;
                        })}
                        <li>
                            <Link to="#" onClick={() => setCurrentPage(p => Math.min(p + 1, Math.ceil(sessions.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(sessions.length / itemsPerPage) ? 'disabled' : ''}>
                                <BsArrowRight size={16} />
                            </Link>
                        </li>
                    </ul>
                </div>
                <CardLoader refreshKey={refreshKey} />
            </div>

            {/* Summary Modal */}
            <Modal show={summaryModalOpen} onHide={closeModal} size="lg">
                <Modal.Header closeButton>
                    <Modal.Title>Session Summary</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {summaryData ? (
                        <>
                            <p><strong>Session ID:</strong> {summaryData.sessionId}</p>
                            <p><strong>Duration:</strong> {summaryData.durationMinutes} minutes</p>
                            <Table striped bordered hover>
                                <thead>
                                    <tr>
                                        <th>Model Type</th>
                                        <th>Total model outs</th>
                                        <th>Cheated</th>
                                        <th>Not cheated</th>
                                        <th>Avg Confidence</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(summaryData.summary).map(([type, data]) => (
                                        <tr key={type}>
                                            <td>{type}</td>
                                            <td>{data.total}</td>
                                            <td>{data.trueCount}</td>
                                            <td>{data.falseCount}</td>
                                            <td>{data.averageConfidence ?? '-'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </Table>

                            {/* New Background Apps Table */}
                                  <h5 className="mt-4">Background Applications Detected</h5>
                                  {summaryData.backgroundApps?.length > 0 ? (
                                    <Table striped bordered hover>
                                      <thead>
                                        <tr>
                                          <th>App Name</th>
                                          <th>Path</th>
                                          <th>First Detected At</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {summaryData.backgroundApps.map((app, idx) => (
                                          <tr key={idx}>
                                            <td>{app.name}</td>
                                            <td style={{ wordBreak: 'break-all', maxWidth: '300px' }}>{app.path}</td>

                                            <td>{new Date(app.firstDetectedAt).toLocaleString()}</td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </Table>
                                  ) : (
                                    <p className="text-muted">No background applications recorded.</p>
                                  )}

                            {/*Copy pasted text */}
                            <h5 className="mt-4">Copied Texts During Exam</h5>
                                    {summaryData.copiedTexts?.length > 0 ? (
                                      <Table striped bordered hover>
                                        <thead>
                                          <tr>
                                            <th>Copied Text</th>
                                            <th>Prediction</th>
                                            <th>Confidence</th>
                                            <th>Timestamp</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {summaryData.copiedTexts.map((item, idx) => {
                                            const pred = (item.modelOutput || '').toLowerCase(); 
                                            const conf =
                                              typeof item.confidence === 'number'
                                                ? item.confidence.toFixed(2)
                                                : '-';
                                        
                                            const badgeClass =
                                              pred === 'ai'
                                                ? 'badge bg-danger'
                                                : pred === 'human'
                                                ? 'badge bg-success'
                                                : 'badge bg-secondary';
                                        
                                            const badgeText = pred || 'unknown';
                                        
                                            return (
                                              <tr key={idx}>
                                                <td style={{ whiteSpace: 'pre-wrap', maxWidth: '600px' }}>{item.text}</td>
                                                <td>
                                                  <span className={badgeClass} style={{ fontSize: '0.85rem' }}>
                                                    {badgeText}
                                                  </span>
                                                </td>
                                                <td>{conf}</td>
                                                <td>{item.timestamp ? new Date(item.timestamp).toLocaleString() : '-'}</td>
                                              </tr>
                                            );
                                          })}
                                        </tbody>
                                      </Table>
                                    ) : (
                                      <p className="text-muted">No copied texts recorded during this session.</p>
                                    )}


                        </>
                    ) : (
                        <p>Loading summary...</p>
                    )}
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={closeModal}>Close</Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
};

export default SubscriberTable;
