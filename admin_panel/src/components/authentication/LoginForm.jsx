import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { loginUser } from '../../contentApi/authApi';
import { setToken } from '../../utils/token';
import Swal from 'sweetalert2';

const LoginForm = ({ registerPath, resetPath }) => {
    const navigate = useNavigate();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMsg, setErrorMsg] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg('');
        setLoading(true);

        Swal.fire({
            title: 'Please wait...',
            allowOutsideClick: false,
            didOpen: () => {
                Swal.showLoading();
            }
        });

        try {
            const { token, user } = await loginUser(email, password);

            // For admin check, if usertype === 99 is admin, else adjust as needed
            if (user.usertype !== 99) {
                Swal.fire({
                    icon: 'error',
                    title: 'Access Denied',
                    text: 'You are not an admin. Please login with an admin account.',
                });
                setLoading(false);
                return;
            }

            // No userName, so use firstname + lastname
            setToken(token, user._id, `${user.firstname} ${user.lastname}`);

            await Swal.fire({
                icon: 'success',
                title: 'Login Successful',
                text: `Welcome back, ${user.firstname || 'User'}!`,
                timer: 1500,
                showConfirmButton: false,
            });

            navigate('/'); 
        } catch (error) {
            Swal.fire({
                icon: 'error',
                title: 'Login Failed',
                text: error.response?.data?.message || 'Invalid credentials',
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <h2 className="fs-20 fw-bolder mb-4">Login</h2>
            <h4 className="fs-13 fw-bold mb-2">Login to your account</h4>
            <p className="fs-12 fw-medium text-muted">
                Thank you for get back <strong>Exam Proctor </strong> Admin application.
            </p>

            <form className="w-100 mt-4 pt-2" onSubmit={handleSubmit}>
                {errorMsg && <div className="alert alert-danger">{errorMsg}</div>}

                <div className="mb-4">
                    <input
                        type="email"
                        className="form-control"
                        placeholder="Email or Username"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                </div>
                <div className="mb-3">
                    <input
                        type="password"
                        className="form-control"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                {/* <div className="d-flex align-items-center justify-content-between">
                    <div>
                        <div className="custom-control custom-checkbox">
                            <input type="checkbox" className="custom-control-input" id="rememberMe" />
                            <label className="custom-control-label c-pointer" htmlFor="rememberMe">Remember Me</label>
                        </div>
                    </div>
                    <div>
                        <Link to={resetPath} className="fs-11 text-primary">Forget password?</Link>
                    </div>
                </div> */}
                <div className="mt-5">
                    <button type="submit" className="btn btn-lg btn-primary w-100">Login</button>
                </div>
            </form>

        </>
    );
};

export default LoginForm;
