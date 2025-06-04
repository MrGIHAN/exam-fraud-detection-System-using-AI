export const setToken = (token, userId, userName) => {
    const expiresAt = new Date().getTime() + 2 * 60 * 60 * 1000;
    localStorage.setItem('token', token);
    localStorage.setItem('expiresAt', expiresAt.toString());
    localStorage.setItem('userId', userId.toString());
    localStorage.setItem('userName', userName || '');
  };

export const getToken = () => {
  const token = localStorage.getItem('token');
  const expiresAt = localStorage.getItem('expiresAt');

  if (!token || !expiresAt) return null;

  const now = new Date().getTime();
  if (now > parseInt(expiresAt)) {
    clearToken();
    return null;
  }

  return token;
};

export const getUserId = () => {
  const id = localStorage.getItem('userId');
  if (!id) return null;
  return id;
};

export const getUserName = () => {
  const userName = localStorage.getItem('userName');
  return userName || null;
};


export const clearToken = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('expiresAt');
    localStorage.removeItem('userId');
    localStorage.removeItem('userName');
  };
