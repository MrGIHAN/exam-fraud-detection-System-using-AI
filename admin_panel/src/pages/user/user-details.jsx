import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import PageHeaderWidgets from '@/components/shared/pageHeader/PageHeaderWidgets';
import Footer from '@/components/shared/Footer';
import UserDetailTable from '@/components/user/UserDetailTable';

const UserDetails = () => {
    return (
        <>
            <PageHeader>
                {/* <PageHeaderWidgets addNewLink="/admin/user/create" name="Add New User" /> */}
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <UserDetailTable title="User details" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default UserDetails;
